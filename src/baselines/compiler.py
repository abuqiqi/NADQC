from abc import ABC, abstractmethod
from typing import Any, Optional
import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import csv

@dataclass
class MappingRecord:
    """
    映射记录类：记录线路层级范围、映射类型、开销及时间
    """
    # 必选字段：线路层级范围
    layer_start: int          # 起始层级（第几层）
    layer_end: int            # 结束层级（第几层）
    # 必选字段：量子比特划分
    partition: list[list[int]]
    # 必选字段：映射信息
    mapping_type: str         # 映射类型（如 "teledata"、"telegate"）
    cost: dict[str, Any]
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None


# 辅助类：管理多条记录
@dataclass
class MappingRecordList:
    """
    映射记录管理器：批量存储、查询记录
    """
    records: list[MappingRecord] = dataclasses.field(default_factory=list)

    def add_record(self, record: MappingRecord):
        """添加一条记录"""
        self.records.append(record)

    def get_records_by_layer_range(self, layer_start: int, layer_end: int) -> list[MappingRecord]:
        """按层级范围查询记录（包含交集）"""
        return [
            r for r in self.records
            if not (r.layer_end < layer_start or r.layer_start > layer_end)
        ]
    
    def save_records(self, filename: str, format: str = "json"):
        """
        将记录保存到文件，支持 JSON/CSV 格式
        
        Args:
            filename: 保存路径（如 "results.json"、"results.csv"）
            format: 保存格式，可选 "json"（默认）、"csv"
        
        Raises:
            ValueError: 不支持的格式
            IOError: 文件写入失败
            TypeError: 数据序列化失败
        """
        if not self.records:
            print("⚠️ 无映射记录可保存")
            return

        # 统一序列化：将 dataclass 转为字典（兼容可选字段 extra_info）
        records_dict = [dataclasses.asdict(record) for record in self.records]

        # 按格式保存
        if format.lower() == "json":
            self._save_as_json(records_dict, filename)
        elif format.lower() == "csv":
            self._save_as_csv(records_dict, filename)
        else:
            raise ValueError(f"不支持的保存格式：{format}，仅支持 json/csv")

    def _save_as_json(self, records_dict: list[dict], filename: str):
        """内部方法：保存为 JSON 格式（兼容 extra_info 字典字段）"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(
                    records_dict,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=False  # 保持字段顺序，更易读
                )
            print(f"✅ 成功保存 {len(records_dict)} 条映射记录到 JSON 文件：{filename}")
        except IOError as e:
            raise IOError(f"❌ JSON 文件写入失败：{str(e)}")
        except TypeError as e:
            raise TypeError(f"❌ 数据序列化失败（检查 extra_info 字段类型）：{str(e)}")

    def _save_as_csv(self, records_dict: list[dict], filename: str):
        """内部方法：保存为 CSV 格式（自动处理可选字段 extra_info）"""
        # 构建表头：必选字段 + extra_info 中的所有键（去重）
        base_fields = [f.name for f in dataclasses.fields(MappingRecord) if f.name != "extra_info"]
        extra_keys = set()
        for record in records_dict:
            if record.get("extra_info"):
                extra_keys.update(record["extra_info"].keys())
        # 最终表头：必选字段在前，extra_info 字段在后
        fieldnames = base_fields + sorted(list(extra_keys))

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # 处理每条记录：合并 extra_info 到主字典，缺失字段填充空值
                for record in records_dict:
                    row = record.copy()
                    extra_info = row.pop("extra_info", {}) or {}  # 取出 extra_info，None 转为空字典
                    row.update(extra_info)  # 合并 extra_info 到主行
                    # 填充缺失的字段（避免 CSV 列数不匹配）
                    for field in fieldnames:
                        if field not in row:
                            row[field] = ""
                    writer.writerow(row)

            print(f"✅ 成功保存 {len(records_dict)} 条映射记录到 CSV 文件：{filename}")
        except IOError as e:
            raise IOError(f"❌ CSV 文件写入失败：{str(e)}")


class Compiler(ABC):
    """
    编译器接口
    """

    @abstractmethod
    def compile(self, circuit: Any, network: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """
        编译电路
        :param circuit: 电路对象
        :param network: 网络对象
        :return: 编译后的电路对象
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        获取编译器名称
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """
        获取映射性能指标
        :return: 包含关键性能指标的字典
        """
        pass


class CompilerFactory:
    """
    编译器工厂类
    """
    _registry = {
        "wbcp": "WBCP"
    }

    @classmethod
    def create_compiler(cls, name: str) -> Compiler:
        """
        创建编译器实例
        :param name: 编译器名称
        :return: 编译器实例
        """
        compiler_type = name.lower()
        if compiler_type not in cls._registry:
            available_compilers = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown compiler: {name}. Available compilers: {available_compilers}")
        
        # 从注册表获取类名，然后创建实例
        compiler_class_name = cls._registry[compiler_type]
        compiler_class = globals()[compiler_class_name]
        return compiler_class()

    @classmethod
    def register_compiler(cls, name: str, compiler_class_name: str):
        """
        注册新的编译器
        :param name: 编译器名称
        :param compiler_class_name: 编译器类名
        """
        cls._registry[name.lower()] = compiler_class_name

    @classmethod
    def unregister_compiler(cls, name: str):
        """
        注销编译器
        :param name: 编译器名称
        """
        compiler_type = name.lower()
        if compiler_type in cls._registry:
            del cls._registry[compiler_type]

    @classmethod
    def get_available_compilers(cls):
        """
        获取可用的编译器列表
        :return: 可用编译器名称列表
        """
        return list(cls._registry.keys())


class BaseCompiler(Compiler):
    """
    基础编译器实现，提供通用功能
    """
    def __init__(self):
        pass

    def evaluate_partitions(self, qig: nx.Graph, partition: list[list[int]], network: Any) -> dict[str, float]:
        """
        计算qubit interaction graph在partitions下的割
        """
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, part in enumerate(partition):
            for node in part:
                node_to_partition[node] = i
        remote_hops, fidelity_loss, fidelity = 0, 0, 1
        for u, v in qig.edges(): # 遍历图中的每一条边，也就是双量子门
            qpu_u = node_to_partition[u]
            qpu_v = node_to_partition[v]
            if qpu_u != qpu_v:
                remote_hops += network.Hops[qpu_u][qpu_v] * qig[u][v]['weight']
                fidelity_loss += (1 - network.W_eff[qpu_u][qpu_v]) * qig[u][v]['weight']
                fidelity *= network.W_eff[qpu_u][qpu_v] ** qig[u][v]['weight']
        return {
            "remote_hops": remote_hops,
            "fidelity_loss": fidelity_loss,
            "fidelity": fidelity
        }
