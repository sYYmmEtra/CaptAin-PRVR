from typing import Dict, Any, List, Union, Type, get_origin, get_args
from pydantic import BaseModel, Field, create_model


class AutoSchema:
    def __init__(self, schema: Union[Dict[str, Any], Type[BaseModel]]):
        self.is_list = False
        if isinstance(schema, type(BaseModel)):
            self.basemodel = schema
        elif isinstance(schema, BaseModel):
            self.basemodel = type(schema)
        else:
            if isinstance(schema, list):
                schema = {
                    'list': schema
                }
                self.is_list = True
            self.basemodel = self.__dict_to_basemodel(schema)


    @staticmethod
    def __dict_to_basemodel(input_dict: Dict[str, Any], class_name: str = "DynamicModel", counter: list = None) -> Type[
        BaseModel]:
        if counter is None:
            counter = [0]
        fields = {}

        def infer_type(value: Any) -> Any:
            if isinstance(value, str):
                return str, Field(description=value)
            elif isinstance(value, int):
                return int, Field(example=value)
            elif isinstance(value, float):
                return float, Field(example=value)
            elif isinstance(value, bool):
                return bool, Field(example=value)
            elif isinstance(value, list):
                if value:
                    item_type = infer_type(value[0])[0]
                    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                        return List[item_type], Field()
                    else:
                        return List[item_type], Field(example=value)
                else:
                    raise ValueError('Empty list is not supported')
            elif isinstance(value, dict):
                counter[0] += 1
                return AutoSchema.__dict_to_basemodel(value, f"{class_name}Sub{counter[0]}", counter), Field()
            else:
                raise ValueError('Unsupported type: {}'.format(type(value)))

        for key, value in input_dict.items():
            fields[key] = infer_type(value)

        dynamic_model = create_model(class_name, **fields, __module__=__name__)
        globals()[dynamic_model.__name__] = dynamic_model   # # 将动态模型添加到全局命名空间,避免pickle序列化时出错
        return dynamic_model

    def set_default_values(self):
        def get_initial_value(field_type):
            if field_type is int:
                return 0
            elif field_type is str:
                return ""
            elif field_type is float:
                return 0.0
            elif field_type is bool:
                return False
            elif get_origin(field_type) is list:    # 递归处理列表,字典和basemodel类型
                element_type = get_args(field_type)[0]
                return [get_initial_value(element_type)]
            elif get_origin(field_type) is dict:
                key_type, value_type = get_args(field_type)
                return {get_initial_value(key_type): get_initial_value(value_type)}
            elif issubclass(field_type, BaseModel):
                field_values = {field_name: get_initial_value(field.annotation)
                                for field_name, field in field_type.__fields__.items()}
                return field_type(**field_values)
            else:
                return None

        fields = self.basemodel.__fields__

        # 根据字段类型动态生成默认值
        initial_data = {
            field_name: get_initial_value(field_info.annotation)
            for field_name, field_info in fields.items()
        }

        # 创建新的 BaseModel 实例
        new_instance = self.basemodel(**initial_data)
        if self.is_list:
            new_instance = new_instance.dict()
            return new_instance['list']
        return new_instance.json()
