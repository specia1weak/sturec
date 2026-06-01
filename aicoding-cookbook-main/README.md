# aicoding-cookbook

[简体中文](#zh-cn) | [English](#english)

<a id="zh-cn"></a>

## 中文说明

`aicoding-cookbook` 是一个面向 AI Coding 工作流的实用资料仓库，收集了本地配置资产、可复用技能和辅助脚本。它的定位不是应用、SDK 或框架，而是一套围绕真实操作场景整理出来的 cookbook，用于配置工具、复用技能，以及标准化日常的 AI 辅助开发流程。

### 适用对象

- 使用 Claude Code、Codex CLI、Augment Context Engine 等 AI 编码工具的开发者
- 希望直接复用现成配置资产，而不是每次从零搭建环境的个人或团队
- 需要维护 Codex 技能、任务跟踪流程和技能打包流程的操作者

### 仓库内容

#### 1. Augment MCP 配置工具包

[`augment-mcp-config/`](./augment-mcp-config) 目录提供了一套面向 Augment Context Engine MCP 的配置资料：

- [`Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md)：中文配置指南，面向 AI 辅助执行场景，覆盖 Augment MCP 和 Codex MCP 在 Windows、macOS、Linux 上的配置流程
- [`augment.mjs`](./augment-mcp-config/augment.mjs)：随指南一起提供的 `augment.mjs` 文件。根据指南说明，这是一个经过修改的 Auggie 入口文件，包含 `codebase-retrieval` 和 `prompt-enhancer` 能力

这个目录适合直接交给 AI 助手执行配置任务，减少手工逐步操作。

#### 2. 可复用的 Codex Skills

[`skills/codex/`](./skills/codex) 目录当前包含 3 个可复用技能包：

- [`taskmaster`](./skills/codex/taskmaster)：多步骤任务跟踪技能，带任务规格、进度日志和基于 CSV 的里程碑管理
- [`todo-list-csv`](./skills/codex/todo-list-csv)：轻量级 CSV 任务跟踪技能，用于把待办文件和 agent 计划保持同步
- [`skill-creator`](./skills/codex/skill-creator)：用于创建、校验和打包新技能的说明与脚本集合

#### 3. 辅助脚本与模板

仓库里不仅有说明文档，也包含可直接复用的脚本和模板：

- [`skills/codex/todo-list-csv/scripts/todo_csv.py`](./skills/codex/todo-list-csv/scripts/todo_csv.py)：Python CLI，用来创建和推进任务 CSV，支持 `path`、`init`、`add`、`start`、`done`、`todo`、`advance`、`plan`、`status`、`cleanup` 等命令
- [`skills/codex/skill-creator/scripts/init_skill.py`](./skills/codex/skill-creator/scripts/init_skill.py)：从模板初始化一个新技能目录
- [`skills/codex/skill-creator/scripts/package_skill.py`](./skills/codex/skill-creator/scripts/package_skill.py)：先校验技能，再把技能目录打包成 zip
- [`skills/codex/skill-creator/scripts/quick_validate.py`](./skills/codex/skill-creator/scripts/quick_validate.py)：对技能 `SKILL.md` 的元数据做轻量校验
- [`skills/codex/taskmaster/assets/`](./skills/codex/taskmaster/assets)：可复用的 `SPEC.md`、`PROGRESS.md`、`TODO.csv` 模板

### 仓库结构

```text
aicoding-cookbook/
├── README.md
├── augment-mcp-config/
│   ├── Augment-MCP配置教程.md
│   └── augment.mjs
└── skills/
    └── codex/
        ├── skill-creator/
        │   ├── LICENSE.txt
        │   ├── SKILL.md
        │   └── scripts/
        ├── taskmaster/
        │   ├── SKILL.md
        │   └── assets/
        └── todo-list-csv/
            ├── SKILL.md
            └── scripts/
```

### 如何使用这个仓库

#### 使用 Augment MCP 配置包

1. 打开 [`augment-mcp-config/`](./augment-mcp-config)
2. 按 [`Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md) 中的步骤执行
3. 只有在你的本地环境与文档假设一致时，才替换已安装环境中的 `augment.mjs`
4. 按文档说明在 Claude Code 或 Codex CLI 中配置对应 MCP 服务

#### 复用 Codex Skills

1. 浏览 [`skills/codex/`](./skills/codex) 下的各个技能目录
2. 阅读每个 `SKILL.md`，确认其触发条件和工作流
3. 如果你需要复用其中的既定流程，优先直接使用随技能附带的脚本和模板

#### 直接复用脚本

如果你只需要自动化部分，也可以直接使用这些脚本：

- `todo_csv.py`：CSV 任务推进
- `init_skill.py`：技能脚手架初始化
- `package_skill.py`：技能打包
- `quick_validate.py`：技能元数据快速校验

### 依赖与前提

- Augment/Auggie 相关配置流程默认需要 Node.js 和 npm
- `skills/codex/` 下的辅助脚本默认需要 Python 3
- Augment 配置指南默认你能够编辑本地工具配置文件，例如 `~/.claude.json` 或 `~/.codex/config.toml`

### 语言说明

- 当前顶层 README 默认使用中文，便于中文读者直接阅读
- 仓库中仍然保留部分中文原始资料，例如 [`augment-mcp-config/Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md)
- 向下滚动即可查看完整英文版说明

### 安全说明

- 不要把真实 API token、私钥或机器相关凭据提交进这个仓库
- 当前 [`.gitignore`](./.gitignore) 已忽略一个本地敏感路径：`augment mcp setting/key.txt`

### 致谢

- [J3n5en](https://github.com/J3n5en) 提供了指南中引用的修改版 `augment.mjs`
- [Augment ACE MCP](https://acemcp.heroman.wtf/)
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli)

<a id="english"></a>

## English

`aicoding-cookbook` is a practical repository of setup assets, reusable skills, and helper scripts for AI coding workflows. It is not an application, SDK, or framework. It is a working cookbook for configuring tools, reusing skill packages, and standardizing day-to-day AI-assisted development routines.

### Who This Repository Is For

- Developers who use AI coding tools such as Claude Code, Codex CLI, and Augment Context Engine
- Individuals or teams who want reusable local setup assets instead of rebuilding toolchains from scratch
- Operators who maintain Codex skills, task-tracking flows, and skill packaging workflows

### What Is Inside

#### 1. Augment MCP configuration kit

The [`augment-mcp-config/`](./augment-mcp-config) directory contains a setup package for Augment Context Engine MCP:

- [`Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md): a Chinese-language setup guide for AI-assisted execution, covering Augment MCP and Codex MCP setup across Windows, macOS, and Linux
- [`augment.mjs`](./augment-mcp-config/augment.mjs): a bundled `augment.mjs` file referenced by the guide. The guide describes it as a modified Auggie entry file with `codebase-retrieval` and `prompt-enhancer` support

This folder is designed to be handed directly to an AI assistant so the assistant can perform most of the setup process.

#### 2. Reusable Codex skills

The [`skills/codex/`](./skills/codex) directory currently contains three reusable skill packages:

- [`taskmaster`](./skills/codex/taskmaster): a multi-step task tracking skill with task specifications, progress logging, and CSV-backed milestone management
- [`todo-list-csv`](./skills/codex/todo-list-csv): a lightweight CSV task tracker that keeps a to-do file synchronized with agent planning
- [`skill-creator`](./skills/codex/skill-creator): a guide and script bundle for creating, validating, and packaging new skills

#### 3. Helper scripts and templates

The repository includes reusable scripts and templates in addition to documentation:

- [`skills/codex/todo-list-csv/scripts/todo_csv.py`](./skills/codex/todo-list-csv/scripts/todo_csv.py): a Python CLI for creating and advancing task CSV files, with commands such as `path`, `init`, `add`, `start`, `done`, `todo`, `advance`, `plan`, `status`, and `cleanup`
- [`skills/codex/skill-creator/scripts/init_skill.py`](./skills/codex/skill-creator/scripts/init_skill.py): initializes a new skill directory from a template
- [`skills/codex/skill-creator/scripts/package_skill.py`](./skills/codex/skill-creator/scripts/package_skill.py): validates and packages a skill directory into a zip archive
- [`skills/codex/skill-creator/scripts/quick_validate.py`](./skills/codex/skill-creator/scripts/quick_validate.py): performs lightweight validation on `SKILL.md` metadata
- [`skills/codex/taskmaster/assets/`](./skills/codex/taskmaster/assets): reusable templates for `SPEC.md`, `PROGRESS.md`, and `TODO.csv`

### Repository Layout

```text
aicoding-cookbook/
├── README.md
├── augment-mcp-config/
│   ├── Augment-MCP配置教程.md
│   └── augment.mjs
└── skills/
    └── codex/
        ├── skill-creator/
        │   ├── LICENSE.txt
        │   ├── SKILL.md
        │   └── scripts/
        ├── taskmaster/
        │   ├── SKILL.md
        │   └── assets/
        └── todo-list-csv/
            ├── SKILL.md
            └── scripts/
```

### How To Use This Repository

#### Use the Augment MCP package

1. Open [`augment-mcp-config/`](./augment-mcp-config)
2. Follow the steps in [`Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md)
3. Replace the installed `augment.mjs` only if your local environment matches the assumptions in the guide
4. Configure the MCP server in Claude Code or Codex CLI as described in the document

#### Reuse the Codex skills

1. Browse the folders under [`skills/codex/`](./skills/codex)
2. Read each `SKILL.md` to understand its trigger conditions and workflow
3. Reuse the bundled scripts and templates when you want the exact packaged workflow

#### Reuse the scripts directly

If you only want the automation pieces, the bundled scripts can also be used directly:

- `todo_csv.py` for CSV-backed task progression
- `init_skill.py` for skill scaffolding
- `package_skill.py` for skill packaging
- `quick_validate.py` for lightweight metadata validation

### Requirements and Assumptions

- Node.js and npm are expected for the Augment/Auggie-related setup flow
- Python 3 is expected for the helper scripts under `skills/codex/`
- The Augment guide assumes you are comfortable editing local tool configuration files such as `~/.claude.json` or `~/.codex/config.toml`

### Language Notes

- This README defaults to Chinese for the primary reading experience
- Some bundled source materials are still Chinese-language documents, especially [`augment-mcp-config/Augment-MCP配置教程.md`](./augment-mcp-config/Augment-MCP配置教程.md)
- The English section mirrors the same repository facts for global readers

### Security Notes

- Do not commit real API tokens, private keys, or machine-specific credentials into this repository
- The current [`.gitignore`](./.gitignore) already excludes a local secret path: `augment mcp setting/key.txt`

### Credits

- [J3n5en](https://github.com/J3n5en) for the modified `augment.mjs` referenced by the bundled guide
- [Augment ACE MCP](https://acemcp.heroman.wtf/)
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli)

- Thanks to the [Linux.do](https://linux.do/) community for project promotion and feedback.
