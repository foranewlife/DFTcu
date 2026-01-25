"""
DFTcu CLI 主入口

提供命令行接口。
"""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="dftcu")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="增加输出详细程度（可多次使用：-v, -vv, -vvv）",
)
@click.pass_context
def cli(ctx, verbose):
    """DFTcu - CUDA 加速的密度泛函理论计算工具

    \b
    示例:
        dftcu pw --config nscf_si.yaml          # 运行 NSCF 计算
        dftcu validate --config nscf_si.yaml    # 验证配置文件
        dftcu convert qe_input.in output.yaml   # 转换 QE 输入文件

    \b
    文档: https://dftcu.readthedocs.io
    """
    # 确保 ctx.obj 存在
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def main():
    """CLI 入口函数"""
    # 导入子命令
    from .pw import pw
    from .validate import validate

    # 注册子命令
    cli.add_command(pw)
    cli.add_command(validate)

    # 运行 CLI
    cli()


if __name__ == "__main__":
    main()
