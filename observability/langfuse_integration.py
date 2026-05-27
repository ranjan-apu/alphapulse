"""
Langfuse integration for LLM observability (Langfuse SDK v3).
Follows Langfuse best practices:
- Nested span hierarchy: trace -> agent_decision -> tool_call spans, llm_generation
- Descriptive trace/span names
- Proper observation types (generation for LLM calls)
- flush() before exit
- Input explicitly set to only relevant data (not all function args)
- Langfuse imported AFTER environment variables are loaded
"""
import os
from typing import Optional, Dict, Any, List

from config import config


class LangfuseTracer:
    """
    Langfuse v3 tracer using start_span / start_observation API.
    Handles the case where Langfuse server is not running (batches locally).

    Trace hierarchy:
      decision_{timestamp} (root span = trace)
        ├── market_state (span): context building
        ├── llm_call (observation 'generation'): each LLM API call
        ├── tool_{name} (span): each tool execution
        └── validation (span): trade validation
    """

    def __init__(self):
        self.enabled = False
        self.langfuse = None

        try:
            # Import AFTER env vars are set (best practice from Langfuse skill)
            os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_BASE_URL"] = config.LANGFUSE_BASE_URL

            from langfuse import Langfuse
            self.langfuse = Langfuse()
            self.enabled = True
            print(f"  [Langfuse] Initialized (host: {config.LANGFUSE_BASE_URL})")

        except Exception as e:
            print(f"  [Langfuse] Initialization failed: {e}")
            print(f"  [Langfuse] Tracing disabled. Continuing without observability.")
            self.enabled = False

    def create_root_span(self, name: str, input_data: Any = None, metadata: Dict = None) -> Optional[Any]:
        """
        Create a root span (effectively a trace).
        Returns a LangfuseSpan that can be used to create child spans.
        """
        if not self.enabled or self.langfuse is None:
            return None

        try:
            span = self.langfuse.start_span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            return span
        except Exception as e:
            print(f"  [Langfuse] create_root_span error: {e}")
            return None

    def add_generation(
        self,
        parent_span: Any,
        name: str,
        model: str,
        input_data: Any,
        output_data: Any = None,
        usage: Dict = None,
        metadata: Dict = None,
    ) -> Optional[Any]:
        """
        Add an LLM generation as a child of parent_span.
        Uses start_observation(as_type='generation') per v3 API best practice.
        """
        if not self.enabled or parent_span is None:
            return None

        try:
            gen = parent_span.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input_data,
                metadata=metadata or {},
            )
            # Set output and usage details via update before end
            if output_data is not None:
                gen.update(output=output_data)
            if usage:
                gen.update(usage_details=usage)
            gen.end()
            return gen
        except Exception as e:
            # Try deprecated start_generation as fallback for older versions
            try:
                gen = parent_span.start_generation(
                    name=name,
                    model=model,
                    input=input_data,
                    metadata=metadata or {},
                )
                if output_data is not None:
                    gen.update(output=output_data)
                if usage:
                    gen.update(usage_details=usage)
                gen.end()
                return gen
            except Exception:
                pass
            return None

    def add_span(
        self,
        parent_span: Any,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Dict = None,
    ) -> Optional[Any]:
        """
        Add a child span under parent_span.
        Used for tool calls, validation, etc.
        """
        if not self.enabled or parent_span is None:
            return None

        try:
            child = parent_span.start_span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            if output_data is not None:
                child.update(output=output_data)
            child.end()
            return child
        except Exception:
            return None

    def end_root_span(self, span: Any, output_data: Any = None, metadata: Dict = None):
        """End a root span, optionally setting output and metadata."""
        if not self.enabled or span is None:
            return

        try:
            if output_data is not None:
                span.update(output=output_data)
            if metadata:
                span.update(metadata=metadata)
            span.end()
        except Exception as e:
            print(f"  [Langfuse] end_root_span error: {e}")

    def flush(self):
        """Flush pending events (MUST call before exit per best practice)."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception:
                pass


def create_tracer() -> LangfuseTracer:
    """Factory function to create a tracer."""
    return LangfuseTracer()
