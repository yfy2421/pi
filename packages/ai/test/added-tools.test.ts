import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { getModel, streamSimple } from "../src/compat.ts";
import type { Api, AssistantMessage, Context, Model, Tool, ToolResultMessage, UserMessage } from "../src/types.ts";
import { estimateContextTokens } from "../src/utils/estimate.ts";

interface AnthropicToolPayload {
	name: string;
	description?: string;
	defer_loading?: boolean;
	cache_control?: { type: string };
}

interface AnthropicMessagePayload {
	role: string;
	content:
		| string
		| Array<{
				type: string;
				id?: string;
				name?: string;
				tool_use_id?: string;
				content?: string | Array<{ type: string; text?: string; tool_name?: string }>;
		  }>;
}

interface AnthropicPayload {
	tools?: AnthropicToolPayload[];
	messages: AnthropicMessagePayload[];
}

class PayloadCaptured extends Error {
	constructor() {
		super("payload captured");
		this.name = "PayloadCaptured";
	}
}

function makeTool(name: string): Tool {
	return {
		name,
		description: `The ${name} tool`,
		parameters: Type.Object({ value: Type.String() }),
	};
}

function makeUserMessage(addedTools?: Tool[]): UserMessage {
	return {
		role: "user",
		content: "Hello",
		...(addedTools ? { addedTools } : {}),
		timestamp: Date.now(),
	};
}

function makeToolResultMessage(toolCallId: string, addedTools?: Tool[]): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "base_tool",
		content: [{ type: "text", text: "done" }],
		...(addedTools ? { addedTools } : {}),
		isError: false,
		timestamp: Date.now(),
	};
}

function makeAssistantToolCall(id: string, name: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "toolCall", id, name, arguments: { value: "x" } }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-opus-4-6",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

/** Context with one prefix tool and one tool added via a tool result. */
function makeToolResultAnchoredContext(baseTools: Tool[], addedTools: Tool[]): Context {
	return {
		messages: [
			makeUserMessage(),
			makeAssistantToolCall("call_1", "base_tool"),
			makeToolResultMessage("call_1", addedTools),
			makeUserMessage(),
		],
		tools: baseTools,
	};
}

async function capturePayload<TPayload>(model: Model<Api>, context: Context, apiKey = "fake-key"): Promise<TPayload> {
	let capturedPayload: TPayload | undefined;

	const payloadCaptureModel = { ...model, baseUrl: "http://127.0.0.1:9" };

	const s = streamSimple(payloadCaptureModel, context, {
		apiKey,
		onPayload: (payload) => {
			capturedPayload = payload as TPayload;
			throw new PayloadCaptured();
		},
	});

	await s.result();

	if (!capturedPayload) {
		throw new Error("Expected payload to be captured before request failure");
	}

	return capturedPayload;
}

function findToolResultBlock(payload: AnthropicPayload): {
	content: Array<{ type: string; text?: string; tool_name?: string }> | string | undefined;
} {
	for (const message of payload.messages) {
		if (typeof message.content === "string") continue;
		for (const block of message.content) {
			if (block.type === "tool_result") {
				return { content: block.content };
			}
		}
	}
	throw new Error("No tool_result block in payload");
}

describe("added tools (Anthropic native path)", () => {
	it("defers tool-result-anchored tools and emits tool_reference blocks", async () => {
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
		const baseTool = payload.tools?.[0];
		const lateTool = payload.tools?.[1];
		expect(baseTool?.defer_loading).toBeUndefined();
		expect(baseTool?.cache_control).toBeDefined();
		expect(lateTool?.defer_loading).toBe(true);
		expect(lateTool?.cache_control).toBeUndefined();

		const toolResult = findToolResultBlock(payload);
		expect(Array.isArray(toolResult.content)).toBe(true);
		const blocks = toolResult.content as Array<{ type: string; tool_name?: string }>;
		expect(blocks.some((b) => b.type === "tool_reference" && b.tool_name === "late_tool")).toBe(true);
	});

	it("defers an active added tool when it has no prior tool use", async () => {
		const lateTool = makeTool("late_tool");
		const context = makeToolResultAnchoredContext([makeTool("base_tool"), lateTool], [lateTool]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.[1]?.defer_loading).toBe(true);
		const toolResult = findToolResultBlock(payload);
		const blocks = toolResult.content as Array<{ type: string; tool_name?: string }>;
		expect(blocks.some((block) => block.type === "tool_reference" && block.tool_name === "late_tool")).toBe(true);
	});

	it("folds an active added tool when the same name was used before its load point", async () => {
		const lateTool = makeTool("late_tool");
		const context: Context = {
			messages: [
				makeUserMessage(),
				makeAssistantToolCall("call_1", "late_tool"),
				makeToolResultMessage("call_1", [lateTool]),
				makeUserMessage(),
			],
			tools: [makeTool("base_tool"), lateTool],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.[1]?.defer_loading).toBeUndefined();
		const toolResult = findToolResultBlock(payload);
		if (Array.isArray(toolResult.content)) {
			expect(toolResult.content.some((block) => block.type === "tool_reference")).toBe(false);
		}
	});

	it("keeps a later user-added tool definition over an earlier deferred definition", async () => {
		const oldLateTool: Tool = { ...makeTool("late_tool"), description: "Old definition" };
		const updatedLateTool: Tool = { ...makeTool("late_tool"), description: "Updated definition" };
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [oldLateTool, makeTool("other_tool")]);
		context.messages[3] = makeUserMessage([updatedLateTool]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool", "other_tool"]);
		const lateTool = payload.tools?.find((tool) => tool.name === "late_tool");
		expect(lateTool?.description).toBe("Updated definition");
		expect(lateTool?.defer_loading).toBeUndefined();
		expect(payload.tools?.find((tool) => tool.name === "other_tool")?.defer_loading).toBe(true);

		const toolResult = findToolResultBlock(payload);
		const blocks = toolResult.content as Array<{ type: string; tool_name?: string }>;
		expect(blocks.some((block) => block.type === "tool_reference" && block.tool_name === "late_tool")).toBe(false);
		expect(blocks.some((block) => block.type === "tool_reference" && block.tool_name === "other_tool")).toBe(true);
	});

	it("folds repeated tool-result redefinitions so historical tool uses stay valid", async () => {
		const oldLateTool: Tool = { ...makeTool("late_tool"), description: "Old definition" };
		const updatedLateTool: Tool = { ...makeTool("late_tool"), description: "Updated definition" };
		const context: Context = {
			messages: [
				makeUserMessage(),
				makeAssistantToolCall("call_1", "base_tool"),
				makeToolResultMessage("call_1", [oldLateTool]),
				makeAssistantToolCall("call_2", "late_tool"),
				makeToolResultMessage("call_2", [updatedLateTool]),
				makeUserMessage(),
			],
			tools: [makeTool("base_tool")],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		const lateTool = payload.tools?.find((tool) => tool.name === "late_tool");
		expect(lateTool?.description).toBe("Updated definition");
		expect(lateTool?.defer_loading).toBeUndefined();

		for (const message of payload.messages) {
			if (typeof message.content === "string") continue;
			for (const block of message.content) {
				if (block.type === "tool_result" && Array.isArray(block.content)) {
					expect(block.content.some((item) => item.type === "tool_reference")).toBe(false);
				}
			}
		}
	});

	it("folds user-anchored tools into the prefix without references", async () => {
		const context: Context = {
			messages: [makeUserMessage([makeTool("late_tool")])],
			tools: [makeTool("base_tool")],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((t) => !t.defer_loading)).toBe(true);
	});

	it("folds everything into the prefix when there is no non-deferred tool", async () => {
		const context = makeToolResultAnchoredContext([], [makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["late_tool"]);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();

		const toolResult = findToolResultBlock(payload);
		if (Array.isArray(toolResult.content)) {
			expect(toolResult.content.some((b) => b.type === "tool_reference")).toBe(false);
		}
	});

	it("falls back to the plain tool list on Haiku", async () => {
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-haiku-4-5"), context);

		expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((t) => !t.defer_loading)).toBe(true);

		const toolResult = findToolResultBlock(payload);
		if (Array.isArray(toolResult.content)) {
			expect(toolResult.content.some((b) => b.type === "tool_reference")).toBe(false);
		}
	});

	it("falls back to the plain tool list on Claude 4.0 IDs", async () => {
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("late_tool")]);
		for (const id of ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]) {
			const model: Model<"anthropic-messages"> = {
				...getModel("anthropic", "claude-opus-4-6"),
				id,
			};
			const payload = await capturePayload<AnthropicPayload>(model, context);

			expect(payload.tools?.map((t) => t.name)).toEqual(["base_tool", "late_tool"]);
			expect(payload.tools?.every((t) => !t.defer_loading)).toBe(true);

			const toolResult = findToolResultBlock(payload);
			if (Array.isArray(toolResult.content)) {
				expect(toolResult.content.some((b) => b.type === "tool_reference")).toBe(false);
			}
		}
	});

	it("respects an explicit supportsToolReferences compat override", async () => {
		const model: Model<"anthropic-messages"> = {
			id: "custom-claude",
			name: "Custom Claude",
			api: "anthropic-messages",
			provider: "vendor-proxy",
			baseUrl: "http://127.0.0.1:9",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 200000,
			maxTokens: 32000,
			compat: { supportsToolReferences: true },
		};
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools?.find((t) => t.name === "late_tool")?.defer_loading).toBe(true);
	});

	it("uses Claude Code canonical names in tool_reference blocks for OAuth tokens", async () => {
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("read")]);
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools?.find((t) => t.name === "Read")?.defer_loading).toBe(true);
		const toolResult = findToolResultBlock(payload);
		const blocks = toolResult.content as Array<{ type: string; tool_name?: string }>;
		expect(blocks.some((b) => b.type === "tool_reference" && b.tool_name === "Read")).toBe(true);
	});

	it("deduplicates base tools after OAuth canonicalization", async () => {
		const context: Context = {
			messages: [makeUserMessage()],
			tools: [makeTool("read"), { ...makeTool("Read"), description: "Uppercase read" }],
		};
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["Read"]);
		expect(payload.tools?.[0]?.description).toBe("Uppercase read");
	});

	it("defers active OAuth-canonicalized added tools with no prior tool use", async () => {
		const context = makeToolResultAnchoredContext(
			[makeTool("bash"), makeTool("read")],
			[{ ...makeTool("Read"), description: "Deferred read" }],
		);
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["Bash", "Read"]);
		expect(payload.tools?.find((tool) => tool.name === "Bash")?.defer_loading).toBeUndefined();
		expect(payload.tools?.find((tool) => tool.name === "Read")?.defer_loading).toBe(true);
		const toolResult = findToolResultBlock(payload);
		const blocks = toolResult.content as Array<{ type: string; tool_name?: string }>;
		expect(blocks.some((block) => block.type === "tool_reference" && block.tool_name === "Read")).toBe(true);
	});

	it("folds a colliding deferred OAuth tool when no non-deferred tool remains", async () => {
		const context = makeToolResultAnchoredContext(
			[makeTool("read")],
			[{ ...makeTool("Read"), description: "Deferred read" }],
		);
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["Read"]);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
		const toolResult = findToolResultBlock(payload);
		if (Array.isArray(toolResult.content)) {
			expect(toolResult.content.some((block) => block.type === "tool_reference")).toBe(false);
		}
	});

	it("keeps case-distinct non-OAuth tools separate", async () => {
		const context: Context = {
			messages: [makeUserMessage()],
			tools: [makeTool("read"), makeTool("Read")],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["read", "Read"]);
	});
});

interface OpenAIToolPayload {
	function?: { name: string };
	name?: string;
}

interface OpenAIToolSearchOutputPayload {
	type: "tool_search_output";
	call_id?: string | null;
	execution?: string;
	status?: string | null;
	tools: Array<{ type: string; name: string; defer_loading?: boolean }>;
}

interface OpenAIToolSearchCallPayload {
	type: "tool_search_call";
	call_id?: string | null;
	execution?: string;
	status?: string | null;
}

interface OpenAIPayload {
	tools?: OpenAIToolPayload[];
	input?: Array<OpenAIToolSearchCallPayload | OpenAIToolSearchOutputPayload | { type?: string }>;
}

function toolNames(payload: OpenAIPayload): string[] {
	return (payload.tools ?? []).map((t) => t.function?.name ?? t.name ?? "");
}

describe("added tools (fallback providers)", () => {
	it("folds added tools into the tool list on openai-completions", async () => {
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [makeTool("late_tool")]);
		const payload = await capturePayload<OpenAIPayload>(getModel("groq", "llama-3.3-70b-versatile"), context);

		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
	});

	it("loads active tool-result-anchored tools through OpenAI tool_search_output", async () => {
		const lateTool = makeTool("late_tool");
		const context = makeToolResultAnchoredContext([makeTool("base_tool"), lateTool], [lateTool]);
		const payload = await capturePayload<OpenAIPayload>(getModel("openai", "gpt-5.2"), context);

		expect(toolNames(payload)).toEqual(["base_tool"]);
		const searchCall = payload.input?.find(
			(item): item is OpenAIToolSearchCallPayload => item.type === "tool_search_call",
		);
		const searchOutput = payload.input?.find(
			(item): item is OpenAIToolSearchOutputPayload => item.type === "tool_search_output",
		);
		expect(searchCall?.execution).toBe("client");
		expect(searchCall?.status).toBe("completed");
		expect(searchOutput?.call_id).toBe(searchCall?.call_id);
		expect(searchOutput?.execution).toBe("client");
		expect(searchOutput?.status).toBe("completed");
		const loadedTool = searchOutput?.tools.find((tool) => tool.name === "late_tool");
		expect(loadedTool).toMatchObject({ type: "function", name: "late_tool", defer_loading: true });
	});

	it("overrides same-named base tools with the added definition", async () => {
		const updated: Tool = { ...makeTool("base_tool"), description: "Updated definition" };
		const context = makeToolResultAnchoredContext([makeTool("base_tool")], [updated]);
		const payload = await capturePayload<{ tools?: { name: string; description: string }[] }>(
			getModel("openai", "gpt-5.2"),
			context,
		);

		expect(payload.tools?.length).toBe(1);
		expect(payload.tools?.[0]?.description).toBe("Updated definition");
	});
});

describe("added tools (context estimation)", () => {
	it("counts tools added after the last usage checkpoint", () => {
		const assistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "done" }],
			api: "openai-responses",
			provider: "openai",
			model: "gpt-5.2",
			usage: {
				input: 50,
				output: 50,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 100,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};
		const plain = estimateContextTokens({ messages: [assistant, makeUserMessage()], tools: [] });
		const largeTool: Tool = { ...makeTool("late_tool"), description: "x".repeat(4000) };
		const withAddedTool = estimateContextTokens({ messages: [assistant, makeUserMessage([largeTool])], tools: [] });

		expect(withAddedTool.lastUsageIndex).toBe(0);
		expect(withAddedTool.tokens).toBeGreaterThan(plain.tokens + 500);
		expect(withAddedTool.trailingTokens).toBeGreaterThan(plain.trailingTokens + 500);
	});
});
