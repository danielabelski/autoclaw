import OpenAI from 'openai';
import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { getToolDefinitions, executeToolHandler } from './tools/index.js';

export class Agent {
  private client: OpenAI;
  private messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
  private model: string;
  private config: any;
  public lastOutputFile: string | null = null;

  constructor(apiKey: string, baseURL: string | undefined, model: string = 'gpt-4-turbo-preview', config: any = {}) {
    this.client = new OpenAI({
      apiKey: apiKey,
      baseURL: baseURL
    });
    this.model = model;
    this.config = config;

    const systemInfo = `
System Information:
- OS: ${os.type()} ${os.release()} (${os.platform()})
- Architecture: ${os.arch()}
- Node.js Version: ${process.version}
- Current Working Directory: ${process.cwd()}
- User: ${os.userInfo().username}
- Home Directory: ${os.homedir()}
- Current Date: ${new Date().toLocaleString()}
`;

    this.messages = [
      {
        role: "system",
        content: `You are AutoClaw, an engineering-first headless agent framework designed for stable, scalable automation.
You operate through precise command-driven execution rather than visual interpretation, ensuring deterministic and reproducible outcomes.
You may run on a local workstation, a headless server, inside a Docker container, or as part of a larger automated pipeline.

CONTEXT:
${systemInfo}

ENVIRONMENT CONSTRAINTS:
1. HEADLESS: No GUI available. Do not try to open browsers or apps.
2. CONTAINER-OPTIMIZED: Assume you are in a sandbox. You can be aggressive with file creation but robust with errors.
3. NON-INTERACTIVE: Always use flags to suppress prompts (e.g., 'apt-get -y', 'rm -rf').

GUIDELINES:
1. EFFICIENCY: Your goal is speed and success. Write scripts that just work.
2. ROBUSTNESS: Use standard Linux/Unix tools found in minimal images (Alpine/Debian).
3. TOOLS: Use 'execute_shell_command' for actions, 'write_file' for code generation.
4. CLARITY: Output concise logs. You are a worker unit, not a chat bot.
5. OPTIMIZATION: When asked to generate creative content (images, stories, complex code), use 'optimize_prompt' first to ensure the best possible output quality.
`
      }
    ];
  }

  async chat(userInput: string): Promise<void> {
    this.messages.push({ role: "user", content: userInput });

    let active = true;
    while (active) {
      const spinner = ora('Thinking...').start();

      try {
        const stream = await this.client.chat.completions.create({
            model: this.model,
            messages: this.messages,
            tools: getToolDefinitions() as any,
            tool_choice: "auto",
            stream: true
        });

        let content = '';
        let reasoningContent = '';
        let toolCalls: { id: string; type: 'function'; function: { name: string; arguments: string } }[] = [];
        let contentStarted = false;
        let reasoningStarted = false;
        const toolNamesSeen = new Set<number>();

        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta as any;

          // Handle reasoning/thinking content (e.g., DeepSeek)
          if (delta?.reasoning_content) {
            if (!reasoningStarted) {
              spinner.stop();
              process.stdout.write(chalk.dim('\n[Thinking] '));
              reasoningStarted = true;
            }
            process.stdout.write(chalk.dim(delta.reasoning_content));
            reasoningContent += delta.reasoning_content;
          }

          // Handle regular content
          if (delta?.content) {
            if (!contentStarted) {
              spinner.stop();
              if (reasoningStarted) process.stdout.write('\n');
              process.stdout.write(chalk.blue("AutoClaw: "));
              contentStarted = true;
            }
            process.stdout.write(delta.content);
            content += delta.content;
          }

          // Handle tool calls - show name as soon as available
          if (delta?.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index;
              if (!toolCalls[idx]) {
                toolCalls[idx] = { id: tc.id || '', type: 'function', function: { name: '', arguments: '' } };
              }
              if (tc.id) toolCalls[idx].id = tc.id;
              if (tc.function?.name) toolCalls[idx].function.name += tc.function.name;
              if (tc.function?.arguments) toolCalls[idx].function.arguments += tc.function.arguments;

              // Show tool name as soon as it's complete
              if (tc.function?.name && !toolNamesSeen.has(idx)) {
                toolNamesSeen.add(idx);
                spinner.stop();
                if (contentStarted) process.stdout.write('\n');
                if (reasoningStarted && !contentStarted) process.stdout.write('\n');
                process.stdout.write(chalk.cyan(`[Calling] ${tc.function.name}\n`));
              }
            }
          }
        }

        if (reasoningStarted) {
          console.log(); // newline after reasoning
        }
        if (contentStarted) {
          console.log(); // newline after streamed content
        }
        if (!reasoningStarted && !contentStarted) {
          spinner.stop();
        }

        // Build the full message for history
        const message: any = { role: "assistant" };
        if (content) message.content = content;
        if (reasoningContent) message.reasoning_content = reasoningContent;
        if (toolCalls.length > 0) {
          message.tool_calls = toolCalls;
          message.content = message.content || null;
        }
        this.messages.push(message);

        if (toolCalls.length > 0) {
          for (const toolCall of toolCalls) {
            if (toolCall.type !== 'function') continue;

            const functionName = toolCall.function.name;
            const functionArgs = JSON.parse(toolCall.function.arguments);

            // Display tool call info
            console.log(chalk.cyan(`\n[Tool] ${functionName}`));
            const argsStr = JSON.stringify(functionArgs, null, 2);
            const argsLines = argsStr.split('\n');
            if (argsLines.length > 8) {
              console.log(chalk.dim(argsLines.slice(0, 8).join('\n')));
              console.log(chalk.dim(`  ... (${argsLines.length - 8} more lines)`));
            } else {
              console.log(chalk.dim(argsStr));
            }

            const execSpinner = ora('Executing...').start();
            let toolResult: string;
            try {
              toolResult = await executeToolHandler(functionName, functionArgs, this.config);
              execSpinner.stop();
            } catch (err: any) {
              execSpinner.fail('Tool execution failed');
              toolResult = `Error: ${err.message}`;
            }

            // Display result with folding for long output
            const MAX_PREVIEW_LINES = 20;
            const resultLines = toolResult.split('\n');

            console.log(chalk.green(`[Result]`));

            if (resultLines.length > MAX_PREVIEW_LINES) {
              // Show preview
              console.log(resultLines.slice(0, MAX_PREVIEW_LINES).join('\n'));
              const remaining = resultLines.length - MAX_PREVIEW_LINES;
              console.log(chalk.dim(`\n  ... ${remaining} more lines (${resultLines.length} lines total)`));

              // Save full output to file
              const outputDir = path.join(os.homedir(), '.autoclaw', 'output');
              if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
              }
              const ts = new Date().toISOString().replace(/[:.]/g, '-');
              const outputFile = path.join(outputDir, `${functionName}_${ts}.txt`);
              fs.writeFileSync(outputFile, toolResult, 'utf-8');
              this.lastOutputFile = outputFile;
              console.log(chalk.dim(`  Type '/view' to see full output`));
            } else {
              console.log(toolResult);
              this.lastOutputFile = null;
            }

            this.messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: toolResult
            });
          }
        } else {
          active = false;
        }

      } catch (error: any) {
        spinner.fail('Error during processing');
        console.error(chalk.red(error.message));
        active = false;
      }
    }
  }
}
