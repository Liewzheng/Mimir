"""Built-in synthetic evaluation datasets for Mimir."""

FRUIT_THEME = [
    "苹果富含维生素 C",
    "香蕉是钾的良好来源",
    "橙子味道酸甜",
    "葡萄可以直接食用",
    "草莓是红色的浆果",
]

CODE_THEME = [
    "Python 支持异步编程",
    "Rust 的所有权系统保证内存安全",
    "函数式编程强调不可变性",
    "单元测试能提高代码质量",
    "Docker 容器化简化了部署",
]

HISTORY_THEME = [
    "秦始皇统一了六国",
    "唐朝是中国古代强盛的朝代",
    "丝绸之路连接了东西方",
    "明朝郑和下西洋",
    "清朝末年发生了鸦片战争",
]


def all_themes() -> list[list[str]]:
    """Return all built-in evaluation themes."""
    return [FRUIT_THEME, CODE_THEME, HISTORY_THEME]
