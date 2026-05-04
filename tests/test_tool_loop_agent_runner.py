import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import ImageURLPart, Message, TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.exceptions import EmptyModelOutputError
from astrbot.core.provider.entities import (
    LLMResponse,
    ProviderRequest,
    ProviderType,
    TokenUsage,
)
from astrbot.core.provider.provider import Provider


class MockProvider(Provider):
    """模拟Provider用于测试"""

    def __init__(self):
        super().__init__({}, {})
        self.call_count = 0
        self.should_call_tools = True
        self.max_calls_before_normal_response = 10

    def get_current_key(self) -> str:
        return "test_key"

    def set_key(self, key: str):
        pass

    async def get_models(self) -> list[str]:
        return ["test_model"]

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1

        # 检查工具是否被禁用
        func_tool = kwargs.get("func_tool")

        # 如果工具被禁用或超过最大调用次数，返回正常响应
        if func_tool is None or self.call_count > self.max_calls_before_normal_response:
            return LLMResponse(
                role="assistant",
                completion_text="这是我的最终回答",
                usage=TokenUsage(input_other=10, output=5),
            )

        # 模拟工具调用响应
        if self.should_call_tools:
            return LLMResponse(
                role="assistant",
                completion_text="我需要使用工具来帮助您",
                tools_call_name=["test_tool"],
                tools_call_args=[{"query": "test"}],
                tools_call_ids=["call_123"],
                usage=TokenUsage(input_other=10, output=5),
            )

        # 默认返回正常响应
        return LLMResponse(
            role="assistant",
            completion_text="这是我的最终回答",
            usage=TokenUsage(input_other=10, output=5),
        )

    async def text_chat_stream(self, **kwargs):
        response = await self.text_chat(**kwargs)
        response.is_chunk = True
        yield response
        response.is_chunk = False
        yield response


class MockToolExecutor:
    """模拟工具执行器"""

    @classmethod
    def execute(cls, tool, run_context, **tool_args):
        async def generator():
            # 模拟工具返回结果，使用正确的类型
            from mcp.types import CallToolResult, TextContent

            result = CallToolResult(
                content=[TextContent(type="text", text="工具执行结果")]
            )
            yield result

        return generator()


class LargeTextToolExecutor:
    """模拟返回超长文本的工具执行器"""

    def __init__(self, text: str):
        self.text = text

    @classmethod
    def from_text(cls, text: str) -> "LargeTextToolExecutor":
        return cls(text)

    def execute(self, tool, run_context, **tool_args):
        async def generator():
            from mcp.types import CallToolResult, TextContent

            result = CallToolResult(content=[TextContent(type="text", text=self.text)])
            yield result

        return generator()


class MockMixedContentToolExecutor:
    """模拟返回图片 + 文本的工具执行器"""

    @classmethod
    def execute(cls, tool, run_context, **tool_args):
        async def generator():
            from mcp.types import CallToolResult, ImageContent, TextContent

            result = CallToolResult(
                content=[
                    ImageContent(
                        type="image",
                        data="dGVzdA==",
                        mimeType="image/png",
                    ),
                    TextContent(type="text", text="直播间标题：新游首发：零~红蝶~"),
                ]
            )
            yield result

        return generator()


class MockFailingProvider(MockProvider):
    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        raise RuntimeError("primary provider failed")


class MockErrProvider(MockProvider):
    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            role="err",
            completion_text="primary provider returned error",
        )


class CapturingProvider(MockProvider):
    def __init__(self, modalities: list[str]):
        super().__init__()
        self.provider_config["modalities"] = modalities
        self.received_contexts = []
        self.received_func_tools = []
        self.should_call_tools = False

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.received_contexts.append(kwargs.get("contexts"))
        self.received_func_tools.append(kwargs.get("func_tool"))
        return LLMResponse(
            role="assistant",
            completion_text="final",
            usage=TokenUsage(input_other=10, output=5),
        )


class MockEmptyOutputThenSuccessProvider(MockProvider):
    def __init__(self, failures_before_success: int = 1):
        super().__init__()
        self.failures_before_success = failures_before_success

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.failures_before_success:
            raise EmptyModelOutputError("model returned no usable output")
        return LLMResponse(
            role="assistant",
            completion_text="这是重试后的最终回答",
            usage=TokenUsage(input_other=10, output=5),
        )


class MockAbortableStreamProvider(MockProvider):
    async def text_chat_stream(self, **kwargs):
        abort_signal = kwargs.get("abort_signal")
        yield LLMResponse(
            role="assistant",
            completion_text="partial ",
            is_chunk=True,
        )
        if abort_signal and abort_signal.is_set():
            yield LLMResponse(
                role="assistant",
                completion_text="partial ",
                is_chunk=False,
            )
            return
        yield LLMResponse(
            role="assistant",
            completion_text="partial final",
            is_chunk=False,
        )


class MockFailingAfterChunkStreamProvider(MockProvider):
    async def text_chat_stream(self, **kwargs):
        self.call_count += 1
        yield LLMResponse(
            role="assistant",
            completion_text="partial ",
            is_chunk=True,
        )
        raise RuntimeError("fallback stream failed")


class MockSuccessStreamProvider(MockProvider):
    """流式 provider，完整成功，用于测试流式 fallback 持久化"""

    async def text_chat_stream(self, **kwargs):
        self.call_count += 1
        yield LLMResponse(
            role="assistant",
            completion_text="partial ",
            is_chunk=True,
        )
        yield LLMResponse(
            role="assistant",
            completion_text="final",
            is_chunk=False,
        )


class MockToolCallProvider(MockProvider):
    def __init__(self, tool_name: str, tool_args: dict[str, str] | None = None):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.abort_signal = None

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.abort_signal = kwargs.get("abort_signal")
        return LLMResponse(
            role="assistant",
            completion_text="",
            tools_call_name=[self.tool_name],
            tools_call_args=[self.tool_args],
            tools_call_ids=[f"call_{self.tool_name}"],
            usage=TokenUsage(input_other=10, output=5),
        )


class SingleToolThenFinalProvider(MockProvider):
    def __init__(self, tool_name: str, tool_args: dict[str, str] | None = None):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args or {}

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        func_tool = kwargs.get("func_tool")
        if func_tool is None or self.call_count > 1:
            return LLMResponse(
                role="assistant",
                completion_text="最终回复",
                usage=TokenUsage(input_other=10, output=5),
            )

        return LLMResponse(
            role="assistant",
            completion_text="",
            tools_call_name=[self.tool_name],
            tools_call_args=[self.tool_args],
            tools_call_ids=["call_large_result"],
            usage=TokenUsage(input_other=10, output=5),
        )


class SequentialToolProvider(MockProvider):
    def __init__(self, tool_sequence: list[str]):
        super().__init__()
        self.tool_sequence = tool_sequence

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        func_tool = kwargs.get("func_tool")
        if func_tool is None or self.call_count > len(self.tool_sequence):
            return LLMResponse(
                role="assistant",
                completion_text="这是我的最终回答",
                usage=TokenUsage(input_other=10, output=5),
            )

        tool_name = self.tool_sequence[self.call_count - 1]
        return LLMResponse(
            role="assistant",
            completion_text="",
            tools_call_name=[tool_name],
            tools_call_args=[{"query": f"step-{self.call_count}"}],
            tools_call_ids=[f"call_{self.call_count}"],
            usage=TokenUsage(input_other=10, output=5),
        )


class MockHandoffProvider(MockToolCallProvider):
    def __init__(self, handoff_tool_name: str):
        super().__init__(handoff_tool_name, {"input": "delegate this task"})


class MockHooks(BaseAgentRunHooks):
    """模拟钩子函数"""

    def __init__(self):
        self.agent_begin_called = False
        self.agent_done_called = False
        self.tool_start_called = False
        self.tool_end_called = False

    async def on_agent_begin(self, run_context):
        self.agent_begin_called = True

    async def on_tool_start(self, run_context, tool, tool_args):
        self.tool_start_called = True

    async def on_tool_end(self, run_context, tool, tool_args, tool_result):
        self.tool_end_called = True

    async def on_agent_done(self, run_context, llm_response):
        self.agent_done_called = True


class MockEvent:
    def __init__(self, umo: str, sender_id: str):
        self.unified_msg_origin = umo
        self._sender_id = sender_id

    def get_sender_id(self):
        return self._sender_id


class MockAgentContext:
    def __init__(self, event):
        self.event = event


class BlockingSubagentContext:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def get_current_chat_provider_id(self, _umo: str) -> str:
        return "provider-id"

    def get_config(self, **_kwargs):
        return {"provider_settings": {}}

    async def tool_loop_agent(self, **_kwargs):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class BlockingToolState:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def handler(self, event, query: str = ""):
        del event, query
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def mock_tool_executor():
    return MockToolExecutor()


@pytest.fixture
def mock_hooks():
    return MockHooks()


@pytest.fixture
def tool_set():
    """创建测试用的工具集"""
    tool = FunctionTool(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=AsyncMock(),
    )
    return ToolSet(tools=[tool])


@pytest.fixture
def provider_request(tool_set):
    """创建测试用的ProviderRequest"""
    return ProviderRequest(prompt="请帮我查询信息", func_tool=tool_set, contexts=[])


@pytest.fixture
def runner():
    """创建ToolLoopAgentRunner实例"""
    return ToolLoopAgentRunner()


def _make_large_tool_result_text() -> str:
    return "x" * 100000


async def _next_agent_response(step_iter):
    return await step_iter.__anext__()


@pytest.mark.asyncio
async def test_max_step_limit_functionality(
    runner, mock_provider, provider_request, mock_tool_executor, mock_hooks
):
    """测试最大步数限制功能"""

    # 设置模拟provider，让它总是返回工具调用
    mock_provider.should_call_tools = True
    mock_provider.max_calls_before_normal_response = (
        100  # 设置一个很大的值，确保不会自然结束
    )

    # 初始化runner
    await runner.reset(
        provider=mock_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    # 设置较小的最大步数来测试限制功能
    max_steps = 3

    # 收集所有响应
    responses = []
    async for response in runner.step_until_done(max_steps):
        responses.append(response)

    # 验证结果
    assert runner.done(), "代理应该在达到最大步数后完成"

    # 验证工具被禁用（这是最重要的验证点）
    assert runner.req.func_tool is None, "达到最大步数后工具应该被禁用"

    # 验证有最终响应
    final_responses = [r for r in responses if r.type == "llm_result"]
    assert len(final_responses) > 0, "应该有最终的LLM响应"

    # 验证最后一条消息是assistant的最终回答
    last_message = runner.run_context.messages[-1]
    assert last_message.role == "assistant", "最后一条消息应该是assistant的最终回答"


@pytest.mark.asyncio
async def test_normal_completion_without_max_step(
    runner, mock_provider, provider_request, mock_tool_executor, mock_hooks
):
    """测试正常完成（不触发最大步数限制）"""

    # 设置模拟provider，让它在第2次调用时返回正常响应
    mock_provider.should_call_tools = True
    mock_provider.max_calls_before_normal_response = 2

    # 初始化runner
    await runner.reset(
        provider=mock_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    # 设置足够大的最大步数
    max_steps = 10

    # 收集所有响应
    responses = []
    async for response in runner.step_until_done(max_steps):
        responses.append(response)

    # 验证结果
    assert runner.done(), "代理应该正常完成"

    # 验证没有触发最大步数限制 - 通过检查provider调用次数
    # mock_provider在第2次调用后返回正常响应，所以不应该达到max_steps(10)
    assert mock_provider.call_count < max_steps, (
        f"正常完成时调用次数({mock_provider.call_count})应该小于最大步数({max_steps})"
    )

    # 验证没有最大步数警告消息（注意：实际注入的是user角色的消息）
    user_messages = [m for m in runner.run_context.messages if m.role == "user"]
    max_step_messages = [
        m for m in user_messages if "工具调用次数已达到上限" in m.content
    ]
    assert len(max_step_messages) == 0, "正常完成时不应该有步数限制消息"

    # 验证工具仍然可用（没有被禁用）
    assert runner.req.func_tool is not None, "正常完成时工具不应该被禁用"


@pytest.mark.asyncio
async def test_max_step_with_streaming(
    runner, mock_provider, provider_request, mock_tool_executor, mock_hooks
):
    """测试流式响应下的最大步数限制"""

    # 设置模拟provider
    mock_provider.should_call_tools = True
    mock_provider.max_calls_before_normal_response = 100

    # 初始化runner，启用流式响应
    await runner.reset(
        provider=mock_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=True,
    )

    # 设置较小的最大步数
    max_steps = 2

    # 收集所有响应
    responses = []
    async for response in runner.step_until_done(max_steps):
        responses.append(response)

    # 验证结果
    assert runner.done(), "代理应该在达到最大步数后完成"

    # 验证有流式响应
    streaming_responses = [r for r in responses if r.type == "streaming_delta"]
    assert len(streaming_responses) > 0, "应该有流式响应"

    # 验证工具被禁用
    assert runner.req.func_tool is None, "达到最大步数后工具应该被禁用"

    # 验证最后一条消息是assistant的最终回答
    last_message = runner.run_context.messages[-1]
    assert last_message.role == "assistant", "最后一条消息应该是assistant的最终回答"


@pytest.mark.asyncio
async def test_hooks_called_with_max_step(
    runner, mock_provider, provider_request, mock_tool_executor, mock_hooks
):
    """测试达到最大步数时钩子函数是否被正确调用"""

    # 设置模拟provider
    mock_provider.should_call_tools = True
    mock_provider.max_calls_before_normal_response = 100

    # 初始化runner
    await runner.reset(
        provider=mock_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    # 设置较小的最大步数
    max_steps = 2

    # 执行步骤
    async for response in runner.step_until_done(max_steps):
        pass

    # 验证钩子函数被调用
    assert mock_hooks.agent_begin_called, "on_agent_begin应该被调用"
    assert mock_hooks.agent_done_called, "on_agent_done应该被调用"
    assert mock_hooks.tool_start_called, "on_tool_start应该被调用"
    assert mock_hooks.tool_end_called, "on_tool_end应该被调用"


@pytest.mark.asyncio
async def test_tool_result_includes_all_calltoolresult_content(
    runner, mock_provider, provider_request, mock_hooks, monkeypatch
):
    """工具返回多个 content 项时，tool result 应包含全部内容。"""

    from astrbot.core.agent.tool_image_cache import tool_image_cache

    mock_provider.should_call_tools = True
    mock_provider.max_calls_before_normal_response = 1

    saved_images = []

    def fake_save_image(
        base64_data, tool_call_id, tool_name, index=0, mime_type="image/png"
    ):
        saved_images.append(
            {
                "base64_data": base64_data,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "index": index,
                "mime_type": mime_type,
            }
        )
        return SimpleNamespace(file_path=f"/tmp/{tool_call_id}_{index}.png")

    monkeypatch.setattr(tool_image_cache, "save_image", fake_save_image)

    await runner.reset(
        provider=mock_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=MockMixedContentToolExecutor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    async for _ in runner.step_until_done(3):
        pass

    tool_messages = [
        m for m in runner.run_context.messages if getattr(m, "role", None) == "tool"
    ]
    assert len(tool_messages) == 1

    content = str(tool_messages[0].content)
    assert "Image returned and cached at path='/tmp/call_123_0.png'." in content
    assert "直播间标题：新游首发：零~红蝶~" in content
    assert saved_images == [
        {
            "base64_data": "dGVzdA==",
            "tool_call_id": "call_123",
            "tool_name": "test_tool",
            "index": 0,
            "mime_type": "image/png",
        }
    ]


@pytest.mark.asyncio
async def test_runner_replaces_runtime_image_context_before_provider_call(
    runner, provider_request, mock_hooks
):
    provider = CapturingProvider(modalities=["tool_use"])

    await runner.reset(
        provider=provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=MockToolExecutor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    runner.run_context.messages.append(
        Message(
            role="user",
            content=[
                TextPart(text="Review this image"),
                ImageURLPart(
                    image_url=ImageURLPart.ImageURL(
                        url="data:image/png;base64,dGVzdA=="
                    )
                ),
            ],
        )
    )

    async for _ in runner.step_until_done(1):
        pass

    assert provider.received_contexts
    sent_context = provider.received_contexts[0]
    assert sent_context[-1]["content"] == [
        {"type": "text", "text": "Review this image"},
        {"type": "text", "text": "[Image]"},
    ]
    assert len(runner.run_context.messages[-2].content) == 2


@pytest.mark.asyncio
async def test_runner_builds_placeholder_for_unsupported_request_image(
    runner, mock_hooks, tool_set
):
    provider = CapturingProvider(modalities=["tool_use"])
    request = ProviderRequest(
        prompt="Describe it",
        image_urls=["/path/that/should/not/be/read.jpg"],
        func_tool=tool_set,
        contexts=[],
    )

    await runner.reset(
        provider=provider,
        request=request,
        run_context=ContextWrapper(context=None),
        tool_executor=MockToolExecutor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    async for _ in runner.step_until_done(1):
        pass

    sent_context = provider.received_contexts[0]
    assert sent_context[-1]["content"] == [
        {"type": "text", "text": "Describe it"},
        {"type": "text", "text": "[Image]"},
    ]


@pytest.mark.asyncio
async def test_runner_clears_tools_for_provider_without_tool_use(
    runner, provider_request, mock_hooks, mock_tool_executor
):
    provider = CapturingProvider(modalities=["text"])

    await runner.reset(
        provider=provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    async for _ in runner.step_until_done(1):
        pass

    assert provider.received_func_tools == [None]


@pytest.mark.asyncio
async def test_same_tool_consecutive_results_include_escalating_guidance(
    runner, mock_tool_executor, mock_hooks
):
    runner_cls = type(runner)
    total_calls = runner_cls.REPEATED_TOOL_NOTICE_L3_THRESHOLD
    provider = SequentialToolProvider(["test_tool"] * total_calls)
    tool = FunctionTool(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=AsyncMock(),
    )
    request = ProviderRequest(
        prompt="请连续执行工具",
        func_tool=ToolSet(tools=[tool]),
        contexts=[],
    )

    await runner.reset(
        provider=provider,
        request=request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    async for _ in runner.step_until_done(total_calls + 1):
        pass

    tool_messages = [
        m for m in runner.run_context.messages if getattr(m, "role", None) == "tool"
    ]
    assert len(tool_messages) == total_calls

    tool_contents = [str(message.content) for message in tool_messages]
    level_1_notice = runner_cls.REPEATED_TOOL_NOTICE_L1_TEMPLATE.format(
        tool_name="test_tool",
        streak=runner_cls.REPEATED_TOOL_NOTICE_L1_THRESHOLD,
    )
    level_2_notice = runner_cls.REPEATED_TOOL_NOTICE_L2_TEMPLATE.format(
        tool_name="test_tool",
        streak=runner_cls.REPEATED_TOOL_NOTICE_L2_THRESHOLD,
    )
    level_3_notice = runner_cls.REPEATED_TOOL_NOTICE_L3_TEMPLATE.format(
        tool_name="test_tool",
        streak=runner_cls.REPEATED_TOOL_NOTICE_L3_THRESHOLD,
    )

    for streak, content in enumerate(tool_contents, start=1):
        if streak < runner_cls.REPEATED_TOOL_NOTICE_L1_THRESHOLD:
            assert level_1_notice not in content
            assert level_2_notice not in content
            assert level_3_notice not in content
        elif streak < runner_cls.REPEATED_TOOL_NOTICE_L2_THRESHOLD:
            assert level_1_notice in content
            assert level_2_notice not in content
            assert level_3_notice not in content
        elif streak < runner_cls.REPEATED_TOOL_NOTICE_L3_THRESHOLD:
            assert level_1_notice not in content
            assert level_2_notice in content
            assert level_3_notice not in content
        else:
            assert level_1_notice not in content
            assert level_2_notice not in content
            assert level_3_notice in content


@pytest.mark.asyncio
async def test_same_tool_streak_resets_after_switching_tools(
    runner, mock_tool_executor, mock_hooks
):
    runner_cls = type(runner)
    repeated_after_reset = runner_cls.REPEATED_TOOL_NOTICE_L1_THRESHOLD
    provider = SequentialToolProvider(
        ["test_tool", "other_tool", *(["test_tool"] * repeated_after_reset)]
    )
    tool_a = FunctionTool(
        name="test_tool",
        description="测试工具 A",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=AsyncMock(),
    )
    tool_b = FunctionTool(
        name="other_tool",
        description="测试工具 B",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=AsyncMock(),
    )
    request = ProviderRequest(
        prompt="切换工具后再重复",
        func_tool=ToolSet(tools=[tool_a, tool_b]),
        contexts=[],
    )

    await runner.reset(
        provider=provider,
        request=request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
    )

    async for _ in runner.step_until_done(repeated_after_reset + 3):
        pass

    tool_messages = [
        m for m in runner.run_context.messages if getattr(m, "role", None) == "tool"
    ]
    assert len(tool_messages) == repeated_after_reset + 2

    tool_contents = [str(message.content) for message in tool_messages]
    level_1_notice = runner_cls.REPEATED_TOOL_NOTICE_L1_TEMPLATE.format(
        tool_name="test_tool",
        streak=runner_cls.REPEATED_TOOL_NOTICE_L1_THRESHOLD,
    )
    level_2_notice = runner_cls.REPEATED_TOOL_NOTICE_L2_TEMPLATE.format(
        tool_name="test_tool",
        streak=runner_cls.REPEATED_TOOL_NOTICE_L2_THRESHOLD,
    )

    assert level_1_notice not in tool_contents[0]
    assert level_1_notice not in tool_contents[1]
    assert level_2_notice not in tool_contents[0]
    assert level_2_notice not in tool_contents[1]

    repeated_contents = tool_contents[2:]
    for streak_after_reset, content in enumerate(repeated_contents, start=1):
        if streak_after_reset < runner_cls.REPEATED_TOOL_NOTICE_L1_THRESHOLD:
            assert level_1_notice not in content
            assert level_2_notice not in content
        elif streak_after_reset < runner_cls.REPEATED_TOOL_NOTICE_L2_THRESHOLD:
            assert level_1_notice in content
            assert level_2_notice not in content
        else:
            assert level_1_notice not in content
            assert level_2_notice in content


@pytest.mark.asyncio
async def test_fallback_provider_used_when_primary_raises(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    primary_provider = MockFailingProvider()
    fallback_provider = MockProvider()
    fallback_provider.should_call_tools = False

    await runner.reset(
        provider=primary_provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
        fallback_providers=[fallback_provider],
    )

    async for _ in runner.step_until_done(5):
        pass

    final_resp = runner.get_final_llm_resp()
    assert final_resp is not None
    assert final_resp.role == "assistant"
    assert final_resp.completion_text == "这是我的最终回答"
    assert primary_provider.call_count == 1
    assert fallback_provider.call_count == 1


@pytest.mark.asyncio
async def test_successful_fallback_provider_is_persisted_for_session(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    primary_provider = MockFailingProvider()
    primary_provider.provider_config["id"] = "primary"
    fallback_provider = MockProvider()
    fallback_provider.provider_config["id"] = "fallback"
    fallback_provider.should_call_tools = False
    provider_manager = SimpleNamespace(set_provider=AsyncMock())
    plugin_context = SimpleNamespace(provider_manager=provider_manager)
    agent_context = SimpleNamespace(
        context=plugin_context,
        event=MockEvent("umo:test", "sender:test"),
    )

    await runner.reset(
        provider=primary_provider,
        request=provider_request,
        run_context=ContextWrapper(context=agent_context),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
        fallback_providers=[fallback_provider],
    )

    async for _ in runner.step_until_done(5):
        pass

    provider_manager.set_provider.assert_awaited_once_with(
        provider_id="fallback",
        provider_type=ProviderType.CHAT_COMPLETION,
        umo="umo:test",
    )


@pytest.mark.asyncio
async def test_streaming_fallback_provider_is_not_persisted_before_completion(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    primary_provider = MockFailingProvider()
    primary_provider.provider_config["id"] = "primary"
    fallback_provider = MockFailingAfterChunkStreamProvider()
    fallback_provider.provider_config["id"] = "fallback"
    provider_manager = SimpleNamespace(set_provider=AsyncMock())
    plugin_context = SimpleNamespace(provider_manager=provider_manager)
    agent_context = SimpleNamespace(
        context=plugin_context,
        event=MockEvent("umo:test", "sender:test"),
    )

    await runner.reset(
        provider=primary_provider,
        request=provider_request,
        run_context=ContextWrapper(context=agent_context),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=True,
        fallback_providers=[fallback_provider],
    )

    async for _ in runner.step_until_done(5):
        pass

    provider_manager.set_provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_fallback_provider_is_persisted_after_completion(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    """PF-05: 流式 fallback 完整成功后应持久化"""
    primary_provider = MockFailingProvider()
    primary_provider.provider_config["id"] = "primary"
    fallback_provider = MockSuccessStreamProvider()
    fallback_provider.provider_config["id"] = "fallback"
    fallback_provider.should_call_tools = False
    provider_manager = SimpleNamespace(set_provider=AsyncMock())
    plugin_context = SimpleNamespace(provider_manager=provider_manager)
    agent_context = SimpleNamespace(
        context=plugin_context,
        event=MockEvent("umo:test", "sender:test"),
    )

    await runner.reset(
        provider=primary_provider,
        request=provider_request,
        run_context=ContextWrapper(context=agent_context),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=True,
        fallback_providers=[fallback_provider],
    )

    async for _ in runner.step_until_done(5):
        pass

    provider_manager.set_provider.assert_awaited_once_with(
        provider_id="fallback",
        provider_type=ProviderType.CHAT_COMPLETION,
        umo="umo:test",
    )


@pytest.mark.asyncio
async def test_second_fallback_persisted_when_first_also_fails(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    """PF-06: 级联 fallback，第二个成功后持久化第二个"""
    primary = MockFailingProvider()
    primary.provider_config["id"] = "primary"
    fallback1 = MockFailingProvider()
    fallback1.provider_config["id"] = "fallback1"
    fallback2 = MockSuccessStreamProvider()
    fallback2.provider_config["id"] = "fallback2"
    fallback2.should_call_tools = False
    provider_manager = SimpleNamespace(set_provider=AsyncMock())
    plugin_context = SimpleNamespace(provider_manager=provider_manager)
    agent_context = SimpleNamespace(
        context=plugin_context,
        event=MockEvent("umo:test", "sender:test"),
    )

    await runner.reset(
        provider=primary,
        request=provider_request,
        run_context=ContextWrapper(context=agent_context),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
        fallback_providers=[fallback1, fallback2],
    )

    async for _ in runner.step_until_done(5):
        pass

    provider_manager.set_provider.assert_awaited_once_with(
        provider_id="fallback2",
        provider_type=ProviderType.CHAT_COMPLETION,
        umo="umo:test",
    )
    assert primary.call_count == 1
    assert fallback1.call_count == 1
    assert fallback2.call_count == 1


@pytest.mark.asyncio
async def test_no_fallback_persistence_when_no_fallback_configured(
    runner, provider_request, mock_tool_executor, mock_hooks
):
    """PF-07: 无 fallback 配置，主 provider 失败时不应调用 set_provider"""
    primary = MockFailingProvider()
    primary.provider_config["id"] = "primary"
    provider_manager = SimpleNamespace(set_provider=AsyncMock())
    plugin_context = SimpleNamespace(provider_manager=provider_manager)
    agent_context = SimpleNamespace(
        context=plugin_context,
        event=MockEvent("umo:test", "sender:test"),
    )

    await runner.reset(
        provider=primary,
        request=provider_request,
        run_context=ContextWrapper(context=agent_context),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=False,
        fallback_providers=[],
    )

    async for _ in runner.step_until_done(5):
        pass

    provider_manager.set_provider.assert_not_awaited()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
