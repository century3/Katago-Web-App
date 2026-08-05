const http = require('http');
const { spawn } = require('child_process');

const HTTP_PORT = 5001;

class KataGoServer {
    constructor() {
        this.process = null;
        this.commandQueue = [];
        this.isProcessing = false;
        this.buffer = '';
        this.startKataGo();
    }
    
    startKataGo() {
        console.log('启动 KataGo GTP 引擎...');
        
        this.process = spawn('C:\\katago\\katago.exe', [
            'gtp',
            '-config', 'C:\\katago\\default_gtp.cfg',
            '-model', 'C:\\katago\\kata1-b28c512nbt-s12192929536-d5655876072.bin.gz'
        ], {
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        // 处理标准输出
        this.process.stdout.on('data', (data) => {
            this.buffer += data.toString();
            this.processBuffer();
        });
        
        // 处理错误输出
        this.process.stderr.on('data', (data) => {
            console.error(`[KataGo STDERR] ${data.toString().trim()}`);
        });
        
        // 进程退出处理
        this.process.on('close', (code) => {
            console.log(`KataGo 进程退出，代码: ${code}`);
            process.exit(code);
        });
        
        this.process.on('error', (err) => {
            console.error('启动 KataGo 失败:', err.message);
            process.exit(1);
        });
        
        // 等待 KataGo 就绪
        setTimeout(() => {
            console.log('KataGo 准备就绪');
            this.startHttpServer();
        }, 3000);
    }
    
    processBuffer() {
        // 按行分割缓冲区
        const lines = this.buffer.split('\n');
        
        // 保留最后不完整的行
        this.buffer = lines.pop() || '';
        
        // 收集所有非空行
        const nonEmptyLines = lines.filter(line => line.trim() !== '');
        
        if (nonEmptyLines.length === 0) return;
        
        // 打印所有收到的行
        nonEmptyLines.forEach(line => {
            console.log(`[GTP] 收到: ${line.trim()}`);
        });
        
        if (this.commandQueue.length === 0) return;
        
        const currentCommand = this.commandQueue[0];
        const isAnalysisCommand = currentCommand.command && 
                                  currentCommand.command.includes('kata-genmove_analyze');
        
        if (isAnalysisCommand) {
            // 对于分析命令，我们需要检查两种情况：
            // 1. 以 "play " 开头的行
            // 2. 或者以 "= " 开头且包含 "play " 的行
            
            // 首先检查是否有 "play " 行
            const playLineIndex = nonEmptyLines.findIndex(line => line.trim().startsWith('play '));
            
            if (playLineIndex !== -1) {
                // 找到 play 行，命令完成
                const playLine = nonEmptyLines[playLineIndex].trim();
                console.log(`[GTP] 分析命令完成，最终结果: ${playLine}`);
                
                const completedCommand = this.commandQueue.shift();
                completedCommand.resolve('= ' + playLine);
                this.isProcessing = false;
                this.currentResponse = '';
                this.processNextCommand();
                return;
            }
            
            // 如果没有 play 行，检查是否有多行响应且最后一行不是单独的 "="
            // 从日志看，KataGo 先返回 "="，然后返回分析数据，最后返回 "play D4"
            // 所以如果看到单独的 "=" 但没有 play 行，应该继续等待
            
            // 检查是否只有单独的 "="
            if (nonEmptyLines.length === 1 && nonEmptyLines[0].trim() === '=') {
                console.log(`[GTP] 收到分析命令的初始响应 "="，继续等待最终结果...`);
                // 存储当前响应，但继续等待
                if (!this.currentResponse) {
                    this.currentResponse = nonEmptyLines[0];
                }
                return; // 继续等待更多数据
            }
            
            // 检查是否有包含 "play " 的 "= play D4" 格式
            const equalsPlayLine = nonEmptyLines.find(line => {
                const trimmed = line.trim();
                return trimmed.startsWith('= ') && trimmed.includes('play ');
            });
            
            if (equalsPlayLine) {
                console.log(`[GTP] 分析命令完成 (格式2): ${equalsPlayLine.trim()}`);
                const completedCommand = this.commandQueue.shift();
                completedCommand.resolve(equalsPlayLine.trim());
                this.isProcessing = false;
                this.currentResponse = '';
                this.processNextCommand();
                return;
            }
            
            // 如果以上都不满足，但 buffer 已经空了，可能是命令异常结束
            if (this.buffer === '' && nonEmptyLines.length > 0) {
                const lastLine = nonEmptyLines[nonEmptyLines.length - 1].trim();
                if (lastLine === '=' || lastLine.startsWith('=')) {
                    console.log(`[GTP] 分析命令异常结束: ${lastLine}`);
                    const completedCommand = this.commandQueue.shift();
                    completedCommand.resolve(lastLine);
                    this.isProcessing = false;
                    this.currentResponse = '';
                    this.processNextCommand();
                }
            }
        } else {
            // 普通命令：查找以 = 或 ? 开头的行
            const responseLineIndex = nonEmptyLines.findIndex(line => {
                const trimmed = line.trim();
                return trimmed.startsWith('=') || trimmed.startsWith('?');
            });
            
            if (responseLineIndex !== -1) {
                const responseLine = nonEmptyLines[responseLineIndex].trim();
                console.log(`[GTP] 普通命令完成: ${responseLine}`);
                
                const completedCommand = this.commandQueue.shift();
                completedCommand.resolve(responseLine);
                this.isProcessing = false;
                this.currentResponse = '';
                this.processNextCommand();
            }
        }
    }
        
    sendCommand(command) {
        return new Promise((resolve, reject) => {
            const commandObj = {
                command,
                resolve,
                reject,
                timeout: setTimeout(() => {
                    reject(new Error('命令超时'));
                    const index = this.commandQueue.findIndex(cmd => cmd === commandObj);
                    if (index !== -1) {
                        this.commandQueue.splice(index, 1);
                    }
                    this.isProcessing = false;
                    this.processNextCommand();
                }, 10000)
            };
            
            this.commandQueue.push(commandObj);
            
            if (!this.isProcessing) {
                this.processNextCommand();
            }
        });
    }
    
    processNextCommand() {
        if (this.commandQueue.length === 0 || this.isProcessing) {
            return;
        }
        
        this.isProcessing = true;
        const commandObj = this.commandQueue[0];
        
        console.log(`[GTP] 发送: ${commandObj.command}`);
        this.process.stdin.write(commandObj.command + '\n');
    }
    
    startHttpServer() {
        const server = http.createServer(async (req, res) => {
            // 设置 CORS 头
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
            
            // 处理 OPTIONS 预检请求
            if (req.method === 'OPTIONS') {
                res.writeHead(200);
                res.end();
                return;
            }
            
            // 只处理 POST /gtp
            if (req.method !== 'POST' || req.url !== '/gtp') {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Not Found' }));
                return;
            }
            
            let body = '';
            req.on('data', chunk => body += chunk);
            
            req.on('end', async () => {
                try {
                    const data = JSON.parse(body);
                    const command = data.command;
                    
                    if (!command || typeof command !== 'string') {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: '无效的命令' }));
                        return;
                    }
                    
                    console.log(`[HTTP] 收到命令: ${command}`);
                    
                    try {
                        const result = await this.sendCommand(command);
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ 
                            success: true, 
                            result: result 
                        }));
                    } catch (error) {
                        res.writeHead(500, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ error: error.message }));
                    }
                    
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: '无效的 JSON 格式' }));
                }
            });
        });
        
        server.listen(HTTP_PORT, () => {
            console.log(`HTTP GTP 代理运行在 http://localhost:${HTTP_PORT}`);
            console.log('使用 curl 测试:');
            console.log(`  curl -X POST http://localhost:${HTTP_PORT}/gtp \\`);
            console.log(`    -H "Content-Type: application/json" \\`);
            console.log(`    -d '{"command":"protocol_version"}'`);
        });
        
        // 优雅关闭
        process.on('SIGINT', () => {
            console.log('\n正在关闭服务器...');
            if (this.process) {
                this.process.kill();
            }
            server.close();
            process.exit(0);
        });
    }
}

// 启动服务器
new KataGoServer();