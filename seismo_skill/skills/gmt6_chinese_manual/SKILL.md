---
name: gmt6_chinese_manual
description: >-
  当用户需要查找 GMT 6 绘图软件的安装方法、基本命令、颜色系统（RGB/HSV）、版本迁移信息或入门教程时使用。该技能基于 GMT 6 官方中文手册内容，提供简洁准确的指引。
category: generated
keywords:
  - GMT
  - Generic Mapping Tools
  - 地理绘图
  - 中文手册
  - 安装
  - Fedora
  - Ubuntu
  - macOS
  - Windows
  - RGB
  - HSV
  - 版本迁移
  - 入门教程
source: generated
generated_by: seismo_skill_docs_builder
generated_from: seismo_skill/docs/GMT6_中文手册/
generated_at: 2026-05-08T15:30:28
---

# GMT 6 中文手册使用指南


## Purpose

这是由 LLM 从原始文档内容整理出的层级化中文技能包。使用它回答问题或编写代码时，应优先读取 `references/outline.md`，再打开相关章节页；这些章节页已经从 PDF/文档正文转换而来，不需要再读取原 PDF。


## When To Use / 何时使用

- 用户需要在不同操作系统（Linux/Windows/macOS）上安装 GMT 6
- 用户需要了解 RGB 和 HSV 颜色模式及转换
- 用户需要从 GMT 4/5 迁移到 GMT 6
- 用户需要学习 GMT 6 的基础绘图命令和流程
- 用户需要确认 GMT 6 的依赖和编译选项
- 用户需要查找 GMT 6 中文手册中的具体章节或索引


## Workflow / 工作流

- 接收用户问题，判断是否属于 GMT 6 中文手册覆盖范围（安装、颜色、版本迁移、入门）。
- 根据问题类型，定位手册中对应章节：第2章（安装）、第3章（版本迁移）、第4章（入门教程）或颜色部分（108-109页）。
- 提取关键信息，例如具体操作系统的安装命令、颜色转换公式、迁移注意事项。
- 以中文提供清晰、简洁的解答，并附上必要的命令示例或说明。
- 验证输出：确保命令格式正确、路径合理、颜色数值符合 GMT 规范。
- 若问题超出手册范围，说明局限并引导用户查阅官方英文文档。


## Converted Skill References

1. 先打开 `references/outline.md` 判断该问题对应哪些子技能。
2. 再打开相关 `subskills/*.md`，从其中提取概念、参数、命令、流程、注意事项和验证方法。
3. 回答时明确区分“文档已说明”和“文档未说明”。不要编造缺失的命令、参数、实验结果或结论。
4. 如果用户要求编程，实现代码前先根据对应章节写一个最小检查或 mini test。
5. 可查看 `references/builder_audit.md` 了解 Skill Builder Agent 的覆盖范围和薄弱项。

- GMT6_中文手册.pdf

## Validation

- 检查所有命令是否使用了正确的 GMT 6 语法（如 modern mode vs classic mode）。
- 验证颜色值转换（RGB↔HSV）是否与 GMT 内置函数一致。
- 确认安装步骤中包管理器命令针对对应发行版正确（如 dnf、apt、brew）。
- 确保版本迁移建议不遗漏关键变更（如 -O -K 废弃，modern mode 的 -R -J 顺序）。
- 核实所有引用页码与提供的文档索引匹配。

## Example Prompts

- 如何在 Fedora 上安装 GMT 6？
- RGB 和 HSV 颜色模式有哪些区别？在 GMT 中如何转换？
- 从 GMT 5 迁移到 GMT 6 需要注意哪些关键变化？
- GMT 6 的入门教程中介绍了哪些基本命令？
- 在 Ubuntu 上通过 apt 安装 GMT 6 的完整步骤是什么？
- GMT 6 的 classic mode 和 modern mode 有什么区别？
