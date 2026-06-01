# MCP 工具自动配置指南

> **本教程面向 Claude Code AI 执行**
> 用户只需将此文件夹拖拽给 Claude Code，提供必要信息，AI 将自动完成全部配置。

---

## 用户操作

1. 将此文件夹拖拽到 Claude Code
2. 输入配置请求，例如：
   - `帮我配置 Augment MCP，token 是 ace_xxxxxx`
   - `帮我配置 Codex MCP`
3. 等待 AI 自动完成配置和测试

---

## 包含的 MCP 工具

| 工具 | 功能 | 需要 |
|------|------|------|
| **Augment Context Engine** | 代码库语义搜索 + 提示词增强 | API Token |
| **OpenAI Codex** | 调用 GPT-5/Codex 模型执行任务 | Codex CLI 已安装 |

---

# AI 执行指令

## 一、Augment Context Engine 配置

### 步骤 1：安装 Auggie CLI

```bash
npm install -g @augmentcode/auggie@prerelease
```

### 步骤 2：定位安装路径

```bash
npm root -g
```

**跨平台路径说明：**
| 系统 | 典型路径 |
|------|----------|
| Windows | `C:\Users\{用户名}\AppData\Roaming\npm\node_modules` 或 `C:\nvm4w\nodejs\node_modules` |
| macOS | `/usr/local/lib/node_modules` 或 `~/.nvm/versions/node/{version}/lib/node_modules` |
| Linux | `/usr/lib/node_modules` 或 `~/.nvm/versions/node/{version}/lib/node_modules` |

### 步骤 3：备份并替换 augment.mjs

**Windows (Git Bash / PowerShell):**
```bash
# Git Bash
cp "{npm_root}/@augmentcode/auggie/augment.mjs" "{npm_root}/@augmentcode/auggie/augment.mjs.backup"
cp "{本目录}/augment.mjs" "{npm_root}/@augmentcode/auggie/augment.mjs"

# PowerShell
Copy-Item "{npm_root}\@augmentcode\auggie\augment.mjs" "{npm_root}\@augmentcode\auggie\augment.mjs.backup"
Copy-Item "{本目录}\augment.mjs" "{npm_root}\@augmentcode\auggie\augment.mjs"
```

**macOS / Linux:**
```bash
cp "{npm_root}/@augmentcode/auggie/augment.mjs" "{npm_root}/@augmentcode/auggie/augment.mjs.backup"
cp "{本目录}/augment.mjs" "{npm_root}/@augmentcode/auggie/augment.mjs"
```

### 步骤 4：配置 MCP 服务器

根据你使用的 Host 工具，选择其一：

**A) Claude Code（编辑 `~/.claude.json` 的 `mcpServers`）**

```json
{
  "augment-context-engine": {
    "command": "auggie",
    "args": ["--mcp"],
    "type": "stdio",
    "env": {
      "AUGMENT_API_TOKEN": "{用户提供的token}",
      "AUGMENT_API_URL": "https://acemcp.heroman.wtf/relay/"
    }
  }
}
```

**B) Codex CLI（编辑 `~/.codex/config.toml` 的 `[mcp_servers]`）**

```toml
[mcp_servers.augment-context-engine]
command = "auggie"
args = ["--mcp"]

[mcp_servers.augment-context-engine.env]
AUGMENT_API_TOKEN = "{用户提供的token}"
AUGMENT_API_URL = "https://acemcp.heroman.wtf/relay/"
```

**B2) Codex CLI（命令行添加，等价于修改 `config.toml`）**

```bash
codex mcp add augment-context-engine --env AUGMENT_API_TOKEN="{用户提供的token}" --env AUGMENT_API_URL="https://acemcp.heroman.wtf/relay/" -- auggie --mcp
```

### 步骤 5：重启并测试

- Claude Code：重启后执行测试：

```
# 测试 1：codebase-retrieval
mcp__augment-context-engine__codebase-retrieval
参数: {"information_request": "What is this project about?"}

# 测试 2：prompt-enhancer
mcp__augment-context-engine__prompt-enhancer
参数: {"origin_prompt": "优化性能"}
```

- Codex CLI：先执行 `codex mcp list` 确认 `augment-context-engine` 已启用，再在 Codex 对话中调用同名工具（方法名与上面一致）。

---

## 二、OpenAI Codex MCP 配置

### 前提条件

用户需要先安装 Codex CLI：
```bash
npm install -g @openai/codex
```

并完成 Codex 登录认证：
```bash
codex login
```

### 步骤 1：配置 MCP 服务器

读取用户的 `~/.claude.json`，在 `mcpServers` 对象中添加：

**Windows:**
```json
{
  "codex": {
    "command": "cmd",
    "args": ["/c", "codex", "mcp-server"],
    "type": "stdio"
  }
}
```

**macOS / Linux:**
```json
{
  "codex": {
    "command": "codex",
    "args": ["mcp-server"],
    "type": "stdio"
  }
}
```

### 步骤 2：重启并测试

提示用户重启 Claude Code，然后执行测试：

```
# 测试 Codex
mcp__codex__codex
参数: {"prompt": "Say hello", "sandbox": "read-only"}
```

**Codex MCP 暴露的工具：**
| 工具 | 功能 |
|------|------|
| `codex` | 启动新的 Codex 会话 |
| `codex-reply` | 继续已有的 Codex 会话 |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `augment.mjs` | J3n5en 修改版，包含 codebase-retrieval + prompt-enhancer |
| `Augment-MCP配置教程.md` | 本文件 |

## 修改版信息

- 基于 Auggie v0.10.1 (commit eb3f80f3)
- 新增 `prompt-enhancer` 工具：将简单提示词转换为详细结构化版本
- 修改者：J3n5en

## 恢复原版 Augment

```bash
cp "{npm_root}/@augmentcode/auggie/augment.mjs.backup" "{npm_root}/@augmentcode/auggie/augment.mjs"
```

---

## 参考资料

- [Augment ACE MCP](https://acemcp.heroman.wtf/)
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli)
- [Codex MCP Documentation](https://developers.openai.com/codex/mcp)
