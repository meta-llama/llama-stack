from dataclasses import dataclass
from typing import Any, Dict, List

from jinja2 import Template


@dataclass
class PromptTemplate:
    template: str
    data: Dict[str, Any]

    def render(self):
        template = Template(self.template)
        return template.render(self.data)


class PromptTemplateGeneratorBase:
    """
    Base class for prompt template generators.
    """

    def gen(self, *args, **kwargs) -> PromptTemplate:
        raise NotImplementedError()

    def data_examples(self) -> List[Any]:
        raise NotImplementedError()
