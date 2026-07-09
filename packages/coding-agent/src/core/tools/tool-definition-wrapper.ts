import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import type { Tool } from "@earendil-works/pi-ai";
import type { ExtensionContext, ToolDefinition, ToolInfo } from "../extensions/types.ts";

interface ActiveToolTracker {
	getActiveTools(): string[];
	getAllTools(): ToolInfo[];
}

function collectAdditiveActiveTools(
	before: readonly string[],
	after: readonly string[],
	allTools: readonly ToolInfo[],
): Tool[] {
	const beforeNames = new Set(before);
	if (!before.every((name) => after.includes(name))) {
		return [];
	}

	const toolInfos = new Map(allTools.map((tool) => [tool.name, tool]));
	const addedTools: Tool[] = [];
	for (const name of after) {
		if (beforeNames.has(name)) continue;
		const tool = toolInfos.get(name);
		if (!tool) continue;
		addedTools.push({
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		});
	}
	return addedTools;
}

async function executeWithAddedTools<TDetails>(
	definition: ToolDefinition<any, TDetails>,
	ctx: ExtensionContext,
	tracker: ActiveToolTracker,
	toolCallId: string,
	params: unknown,
	signal: AbortSignal | undefined,
	onUpdate: Parameters<AgentTool<any, TDetails>["execute"]>[3],
): Promise<AgentToolResult<TDetails>> {
	const activeBefore = tracker.getActiveTools();
	const result = await definition.execute(toolCallId, params as never, signal, onUpdate, ctx);
	const addedTools = collectAdditiveActiveTools(activeBefore, tracker.getActiveTools(), tracker.getAllTools());
	if (addedTools.length === 0) {
		return result;
	}
	return {
		...result,
		addedTools: [...(result.addedTools ?? []), ...addedTools],
	};
}

/** Wrap a ToolDefinition into an AgentTool for the core runtime. */
export function wrapToolDefinition<TDetails = unknown>(
	definition: ToolDefinition<any, TDetails>,
	ctxFactory?: () => ExtensionContext,
	activeToolTracker?: ActiveToolTracker,
): AgentTool<any, TDetails> {
	return {
		name: definition.name,
		label: definition.label,
		description: definition.description,
		parameters: definition.parameters,
		prepareArguments: definition.prepareArguments,
		executionMode: definition.executionMode,
		execute: (toolCallId, params, signal, onUpdate) => {
			const ctx = ctxFactory?.();
			return ctx && activeToolTracker
				? executeWithAddedTools(definition, ctx, activeToolTracker, toolCallId, params, signal, onUpdate)
				: definition.execute(toolCallId, params, signal, onUpdate, ctx as ExtensionContext);
		},
	};
}

/** Wrap multiple ToolDefinitions into AgentTools for the core runtime. */
export function wrapToolDefinitions(
	definitions: ToolDefinition<any, any>[],
	ctxFactory?: () => ExtensionContext,
	activeToolTracker?: ActiveToolTracker,
): AgentTool<any>[] {
	return definitions.map((definition) => wrapToolDefinition(definition, ctxFactory, activeToolTracker));
}

/**
 * Synthesize a minimal ToolDefinition from an AgentTool.
 *
 * This keeps AgentSession's internal registry definition-first even when a caller
 * provides plain AgentTool overrides that do not include prompt metadata or renderers.
 */
export function createToolDefinitionFromAgentTool(tool: AgentTool<any>): ToolDefinition<any, unknown> {
	return {
		name: tool.name,
		label: tool.label,
		description: tool.description,
		parameters: tool.parameters as any,
		prepareArguments: tool.prepareArguments,
		executionMode: tool.executionMode,
		execute: async (toolCallId, params, signal, onUpdate) => tool.execute(toolCallId, params, signal, onUpdate),
	};
}
