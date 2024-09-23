# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import ipaddress
import typing
from typing import Any, Dict, Set, Union

from ..strong_typing.core import JsonType
from ..strong_typing.docstring import Docstring, parse_type
from ..strong_typing.inspection import (
    is_generic_list,
    is_type_optional,
    is_type_union,
    unwrap_generic_list,
    unwrap_optional_type,
    unwrap_union_types,
)
from ..strong_typing.name import python_type_to_name
from ..strong_typing.schema import (
    get_schema_identifier,
    JsonSchemaGenerator,
    register_schema,
    Schema,
    SchemaOptions,
)
from ..strong_typing.serialization import json_dump_string, object_to_json

from .operations import (
    EndpointOperation,
    get_endpoint_events,
    get_endpoint_operations,
    HTTPMethod,
)
from .options import *
from .specification import (
    Components,
    Document,
    Example,
    ExampleRef,
    MediaType,
    Operation,
    Parameter,
    ParameterLocation,
    PathItem,
    RequestBody,
    Response,
    ResponseRef,
    SchemaOrRef,
    SchemaRef,
    Tag,
    TagGroup,
)

register_schema(
    ipaddress.IPv4Address,
    schema={
        "type": "string",
        "format": "ipv4",
        "title": "IPv4 address",
        "description": "IPv4 address, according to dotted-quad ABNF syntax as defined in RFC 2673, section 3.2.",
    },
    examples=["192.0.2.0", "198.51.100.1", "203.0.113.255"],
)

register_schema(
    ipaddress.IPv6Address,
    schema={
        "type": "string",
        "format": "ipv6",
        "title": "IPv6 address",
        "description": "IPv6 address, as defined in RFC 2373, section 2.2.",
    },
    examples=[
        "FEDC:BA98:7654:3210:FEDC:BA98:7654:3210",
        "1080:0:0:0:8:800:200C:417A",
        "1080::8:800:200C:417A",
        "FF01::101",
        "::1",
    ],
)


def http_status_to_string(status_code: HTTPStatusCode) -> str:
    "Converts an HTTP status code to a string."

    if isinstance(status_code, HTTPStatus):
        return str(status_code.value)
    elif isinstance(status_code, int):
        return str(status_code)
    elif isinstance(status_code, str):
        return status_code
    else:
        raise TypeError("expected: HTTP status code")


class SchemaBuilder:
    schema_generator: JsonSchemaGenerator
    schemas: Dict[str, Schema]

    def __init__(self, schema_generator: JsonSchemaGenerator) -> None:
        self.schema_generator = schema_generator
        self.schemas = {}

    def classdef_to_schema(self, typ: type) -> Schema:
        """
        Converts a type to a JSON schema.
        For nested types found in the type hierarchy, adds the type to the schema registry in the OpenAPI specification section `components`.
        """

        type_schema, type_definitions = self.schema_generator.classdef_to_schema(typ)

        # append schema to list of known schemas, to be used in OpenAPI's Components Object section
        for ref, schema in type_definitions.items():
            self._add_ref(ref, schema)

        return type_schema

    def classdef_to_named_schema(self, name: str, typ: type) -> Schema:
        schema = self.classdef_to_schema(typ)
        self._add_ref(name, schema)
        return schema

    def classdef_to_ref(self, typ: type) -> SchemaOrRef:
        """
        Converts a type to a JSON schema, and if possible, returns a schema reference.
        For composite types (such as classes), adds the type to the schema registry in the OpenAPI specification section `components`.
        """

        type_schema = self.classdef_to_schema(typ)
        if typ is str or typ is int or typ is float:
            # represent simple types as themselves
            return type_schema

        type_name = get_schema_identifier(typ)
        if type_name is not None:
            return self._build_ref(type_name, type_schema)

        try:
            type_name = python_type_to_name(typ)
            return self._build_ref(type_name, type_schema)
        except TypeError:
            pass

        return type_schema

    def _build_ref(self, type_name: str, type_schema: Schema) -> SchemaRef:
        self._add_ref(type_name, type_schema)
        return SchemaRef(type_name)

    def _add_ref(self, type_name: str, type_schema: Schema) -> None:
        if type_name not in self.schemas:
            self.schemas[type_name] = type_schema


class ContentBuilder:
    schema_builder: SchemaBuilder
    schema_transformer: Optional[Callable[[SchemaOrRef], SchemaOrRef]]
    sample_transformer: Optional[Callable[[JsonType], JsonType]]

    def __init__(
        self,
        schema_builder: SchemaBuilder,
        schema_transformer: Optional[Callable[[SchemaOrRef], SchemaOrRef]] = None,
        sample_transformer: Optional[Callable[[JsonType], JsonType]] = None,
    ) -> None:
        self.schema_builder = schema_builder
        self.schema_transformer = schema_transformer
        self.sample_transformer = sample_transformer

    def build_content(
        self, payload_type: type, examples: Optional[List[Any]] = None
    ) -> Dict[str, MediaType]:
        "Creates the content subtree for a request or response."

        if is_generic_list(payload_type):
            media_type = "application/jsonl"
            item_type = unwrap_generic_list(payload_type)
        else:
            media_type = "application/json"
            item_type = payload_type

        return {media_type: self.build_media_type(item_type, examples)}

    def build_media_type(
        self, item_type: type, examples: Optional[List[Any]] = None
    ) -> MediaType:
        schema = self.schema_builder.classdef_to_ref(item_type)
        if self.schema_transformer:
            schema_transformer: Callable[[SchemaOrRef], SchemaOrRef] = self.schema_transformer  # type: ignore
            schema = schema_transformer(schema)

        if not examples:
            return MediaType(schema=schema)

        if len(examples) == 1:
            return MediaType(schema=schema, example=self._build_example(examples[0]))

        return MediaType(
            schema=schema,
            examples=self._build_examples(examples),
        )

    def _build_examples(
        self, examples: List[Any]
    ) -> Dict[str, Union[Example, ExampleRef]]:
        "Creates a set of several examples for a media type."

        if self.sample_transformer:
            sample_transformer: Callable[[JsonType], JsonType] = self.sample_transformer  # type: ignore
        else:
            sample_transformer = lambda sample: sample

        results: Dict[str, Union[Example, ExampleRef]] = {}
        for example in examples:
            value = sample_transformer(object_to_json(example))

            hash_string = (
                hashlib.md5(json_dump_string(value).encode("utf-8")).digest().hex()
            )
            name = f"ex-{hash_string}"

            results[name] = Example(value=value)

        return results

    def _build_example(self, example: Any) -> Any:
        "Creates a single example for a media type."

        if self.sample_transformer:
            sample_transformer: Callable[[JsonType], JsonType] = self.sample_transformer  # type: ignore
        else:
            sample_transformer = lambda sample: sample

        return sample_transformer(object_to_json(example))


@dataclass
class ResponseOptions:
    """
    Configuration options for building a response for an operation.

    :param type_descriptions: Maps each response type to a textual description (if available).
    :param examples: A list of response examples.
    :param status_catalog: Maps each response type to an HTTP status code.
    :param default_status_code: HTTP status code assigned to responses that have no mapping.
    """

    type_descriptions: Dict[type, str]
    examples: Optional[List[Any]]
    status_catalog: Dict[type, HTTPStatusCode]
    default_status_code: HTTPStatusCode


@dataclass
class StatusResponse:
    status_code: str
    types: List[type] = dataclasses.field(default_factory=list)
    examples: List[Any] = dataclasses.field(default_factory=list)


class ResponseBuilder:
    content_builder: ContentBuilder

    def __init__(self, content_builder: ContentBuilder) -> None:
        self.content_builder = content_builder

    def _get_status_responses(
        self, options: ResponseOptions
    ) -> Dict[str, StatusResponse]:
        status_responses: Dict[str, StatusResponse] = {}

        for response_type in options.type_descriptions.keys():
            status_code = http_status_to_string(
                options.status_catalog.get(response_type, options.default_status_code)
            )

            # look up response for status code
            if status_code not in status_responses:
                status_responses[status_code] = StatusResponse(status_code)
            status_response = status_responses[status_code]

            # append response types that are assigned the given status code
            status_response.types.append(response_type)

            # append examples that have the matching response type
            if options.examples:
                status_response.examples.extend(
                    example
                    for example in options.examples
                    if isinstance(example, response_type)
                )

        return dict(sorted(status_responses.items()))

    def build_response(
        self, options: ResponseOptions
    ) -> Dict[str, Union[Response, ResponseRef]]:
        """
        Groups responses that have the same status code.
        """

        responses: Dict[str, Union[Response, ResponseRef]] = {}
        status_responses = self._get_status_responses(options)
        for status_code, status_response in status_responses.items():
            response_types = tuple(status_response.types)
            if len(response_types) > 1:
                composite_response_type: type = Union[response_types]  # type: ignore
            else:
                (response_type,) = response_types
                composite_response_type = response_type

            description = " **OR** ".join(
                filter(
                    None,
                    (
                        options.type_descriptions[response_type]
                        for response_type in response_types
                    ),
                )
            )

            responses[status_code] = self._build_response(
                response_type=composite_response_type,
                description=description,
                examples=status_response.examples or None,
            )

        return responses

    def _build_response(
        self,
        response_type: type,
        description: str,
        examples: Optional[List[Any]] = None,
    ) -> Response:
        "Creates a response subtree."

        if response_type is not None:
            return Response(
                description=description,
                content=self.content_builder.build_content(response_type, examples),
            )
        else:
            return Response(description=description)


def schema_error_wrapper(schema: SchemaOrRef) -> Schema:
    "Wraps an error output schema into a top-level error schema."

    return {
        "type": "object",
        "properties": {
            "error": schema,  # type: ignore
        },
        "additionalProperties": False,
        "required": [
            "error",
        ],
    }


def sample_error_wrapper(error: JsonType) -> JsonType:
    "Wraps an error output sample into a top-level error sample."

    return {"error": error}


class Generator:
    endpoint: type
    options: Options
    schema_builder: SchemaBuilder
    responses: Dict[str, Response]

    def __init__(self, endpoint: type, options: Options) -> None:
        self.endpoint = endpoint
        self.options = options
        schema_generator = JsonSchemaGenerator(
            SchemaOptions(
                definitions_path="#/components/schemas/",
                use_examples=self.options.use_examples,
                property_description_fun=options.property_description_fun,
            )
        )
        self.schema_builder = SchemaBuilder(schema_generator)
        self.responses = {}

    def _build_type_tag(self, ref: str, schema: Schema) -> Tag:
        definition = f'<SchemaDefinition schemaRef="#/components/schemas/{ref}" />'
        title = typing.cast(str, schema.get("title"))
        description = typing.cast(str, schema.get("description"))
        return Tag(
            name=ref,
            description="\n\n".join(
                s for s in (title, description, definition) if s is not None
            ),
        )

    def _build_extra_tag_groups(
        self, extra_types: Dict[str, List[type]]
    ) -> Dict[str, List[Tag]]:
        """
        Creates a dictionary of tag group captions as keys, and tag lists as values.

        :param extra_types: A dictionary of type categories and list of types in that category.
        """

        extra_tags: Dict[str, List[Tag]] = {}

        for category_name, category_items in extra_types.items():
            tag_list: List[Tag] = []

            for extra_type in category_items:
                name = python_type_to_name(extra_type)
                schema = self.schema_builder.classdef_to_named_schema(name, extra_type)
                tag_list.append(self._build_type_tag(name, schema))

            if tag_list:
                extra_tags[category_name] = tag_list

        return extra_tags

    def _build_operation(self, op: EndpointOperation) -> Operation:
        doc_string = parse_type(op.func_ref)
        doc_params = dict(
            (param.name, param.description) for param in doc_string.params.values()
        )

        # parameters passed in URL component path
        path_parameters = [
            Parameter(
                name=param_name,
                in_=ParameterLocation.Path,
                description=doc_params.get(param_name),
                required=True,
                schema=self.schema_builder.classdef_to_ref(param_type),
            )
            for param_name, param_type in op.path_params
        ]

        # parameters passed in URL component query string
        query_parameters = []
        for param_name, param_type in op.query_params:
            if is_type_optional(param_type):
                inner_type: type = unwrap_optional_type(param_type)
                required = False
            else:
                inner_type = param_type
                required = True

            query_parameter = Parameter(
                name=param_name,
                in_=ParameterLocation.Query,
                description=doc_params.get(param_name),
                required=required,
                schema=self.schema_builder.classdef_to_ref(inner_type),
            )
            query_parameters.append(query_parameter)

        # parameters passed anywhere
        parameters = path_parameters + query_parameters
        parameters += [
            Parameter(
                name="X-LlamaStack-ProviderData",
                in_=ParameterLocation.Header,
                description="JSON-encoded provider data which will be made available to the adapter servicing the API",
                required=False,
                schema=self.schema_builder.classdef_to_ref(str),
            )
        ]

        # data passed in payload
        if op.request_params:
            builder = ContentBuilder(self.schema_builder)
            first = next(iter(op.request_params))
            request_name, request_type = first

            from dataclasses import make_dataclass

            op_name = "".join(word.capitalize() for word in op.name.split("_"))
            request_name = f"{op_name}Request"
            request_type = make_dataclass(request_name, op.request_params)

            requestBody = RequestBody(
                content={
                    "application/json": builder.build_media_type(
                        request_type, op.request_examples
                    )
                },
                description=doc_params.get(request_name),
                required=True,
            )
        else:
            requestBody = None

        # success response types
        if doc_string.returns is None and is_type_union(op.response_type):
            # split union of return types into a list of response types
            success_type_docstring: Dict[type, Docstring] = {
                typing.cast(type, item): parse_type(item)
                for item in unwrap_union_types(op.response_type)
            }
            success_type_descriptions = {
                item: doc_string.short_description
                for item, doc_string in success_type_docstring.items()
                if doc_string.short_description
            }
        else:
            # use return type as a single response type
            success_type_descriptions = {
                op.response_type: (
                    doc_string.returns.description if doc_string.returns else "OK"
                )
            }

        response_examples = op.response_examples or []
        success_examples = [
            example
            for example in response_examples
            if not isinstance(example, Exception)
        ]

        content_builder = ContentBuilder(self.schema_builder)
        response_builder = ResponseBuilder(content_builder)
        response_options = ResponseOptions(
            success_type_descriptions,
            success_examples if self.options.use_examples else None,
            self.options.success_responses,
            "200",
        )
        responses = response_builder.build_response(response_options)

        # failure response types
        if doc_string.raises:
            exception_types: Dict[type, str] = {
                item.raise_type: item.description for item in doc_string.raises.values()
            }
            exception_examples = [
                example
                for example in response_examples
                if isinstance(example, Exception)
            ]

            if self.options.error_wrapper:
                schema_transformer = schema_error_wrapper
                sample_transformer = sample_error_wrapper
            else:
                schema_transformer = None
                sample_transformer = None

            content_builder = ContentBuilder(
                self.schema_builder,
                schema_transformer=schema_transformer,
                sample_transformer=sample_transformer,
            )
            response_builder = ResponseBuilder(content_builder)
            response_options = ResponseOptions(
                exception_types,
                exception_examples if self.options.use_examples else None,
                self.options.error_responses,
                "500",
            )
            responses.update(response_builder.build_response(response_options))

        if op.event_type is not None:
            builder = ContentBuilder(self.schema_builder)
            callbacks = {
                f"{op.func_name}_callback": {
                    "{$request.query.callback}": PathItem(
                        post=Operation(
                            requestBody=RequestBody(
                                content=builder.build_content(op.event_type)
                            ),
                            responses={"200": Response(description="OK")},
                        )
                    )
                }
            }

        else:
            callbacks = None

        return Operation(
            tags=[op.defining_class.__name__],
            summary=doc_string.short_description,
            description=doc_string.long_description,
            parameters=parameters,
            requestBody=requestBody,
            responses=responses,
            callbacks=callbacks,
            security=[] if op.public else None,
        )

    def generate(self) -> Document:
        paths: Dict[str, PathItem] = {}
        endpoint_classes: Set[type] = set()
        for op in get_endpoint_operations(
            self.endpoint, use_examples=self.options.use_examples
        ):
            endpoint_classes.add(op.defining_class)

            operation = self._build_operation(op)

            if op.http_method is HTTPMethod.GET:
                pathItem = PathItem(get=operation)
            elif op.http_method is HTTPMethod.PUT:
                pathItem = PathItem(put=operation)
            elif op.http_method is HTTPMethod.POST:
                pathItem = PathItem(post=operation)
            elif op.http_method is HTTPMethod.DELETE:
                pathItem = PathItem(delete=operation)
            elif op.http_method is HTTPMethod.PATCH:
                pathItem = PathItem(patch=operation)
            else:
                raise NotImplementedError(f"unknown HTTP method: {op.http_method}")

            route = op.get_route()
            if route in paths:
                paths[route].update(pathItem)
            else:
                paths[route] = pathItem

        operation_tags: List[Tag] = []
        for cls in endpoint_classes:
            doc_string = parse_type(cls)
            operation_tags.append(
                Tag(
                    name=cls.__name__,
                    description=doc_string.long_description,
                    displayName=doc_string.short_description,
                )
            )

        # types that are produced/consumed by operations
        type_tags = [
            self._build_type_tag(ref, schema)
            for ref, schema in self.schema_builder.schemas.items()
        ]

        # types that are emitted by events
        event_tags: List[Tag] = []
        events = get_endpoint_events(self.endpoint)
        for ref, event_type in events.items():
            event_schema = self.schema_builder.classdef_to_named_schema(ref, event_type)
            event_tags.append(self._build_type_tag(ref, event_schema))

        # types that are explicitly declared
        extra_tag_groups: Dict[str, List[Tag]] = {}
        if self.options.extra_types is not None:
            if isinstance(self.options.extra_types, list):
                extra_tag_groups = self._build_extra_tag_groups(
                    {"AdditionalTypes": self.options.extra_types}
                )
            elif isinstance(self.options.extra_types, dict):
                extra_tag_groups = self._build_extra_tag_groups(
                    self.options.extra_types
                )
            else:
                raise TypeError(
                    f"type mismatch for collection of extra types: {type(self.options.extra_types)}"
                )

        # list all operations and types
        tags: List[Tag] = []
        tags.extend(operation_tags)
        tags.extend(type_tags)
        tags.extend(event_tags)
        for extra_tag_group in extra_tag_groups.values():
            tags.extend(extra_tag_group)

        tag_groups = []
        if operation_tags:
            tag_groups.append(
                TagGroup(
                    name=self.options.map("Operations"),
                    tags=sorted(tag.name for tag in operation_tags),
                )
            )
        if type_tags:
            tag_groups.append(
                TagGroup(
                    name=self.options.map("Types"),
                    tags=sorted(tag.name for tag in type_tags),
                )
            )
        if event_tags:
            tag_groups.append(
                TagGroup(
                    name=self.options.map("Events"),
                    tags=sorted(tag.name for tag in event_tags),
                )
            )
        for caption, extra_tag_group in extra_tag_groups.items():
            tag_groups.append(
                TagGroup(
                    name=self.options.map(caption),
                    tags=sorted(tag.name for tag in extra_tag_group),
                )
            )

        if self.options.default_security_scheme:
            securitySchemes = {"Default": self.options.default_security_scheme}
        else:
            securitySchemes = None

        return Document(
            openapi=".".join(str(item) for item in self.options.version),
            info=self.options.info,
            jsonSchemaDialect=(
                "https://json-schema.org/draft/2020-12/schema"
                if self.options.version >= (3, 1, 0)
                else None
            ),
            servers=[self.options.server],
            paths=paths,
            components=Components(
                schemas=self.schema_builder.schemas,
                responses=self.responses,
                securitySchemes=securitySchemes,
            ),
            security=[{"Default": []}],
            tags=tags,
            tagGroups=tag_groups,
        )
