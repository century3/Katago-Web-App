const http = require('http');
const fs = require('fs');
const path = require('path');

const BASE_DIR = process.cwd();

const envPort = parseInt(process.env.PORT, 10);
let port = Number.isFinite(envPort) && envPort > 0 ? envPort : 3000;
const portMax = port + 32;

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
    
    const reqUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    let pathname = decodeURIComponent(reqUrl.pathname);

    if (pathname === '/' || pathname === '') pathname = '/go_board.html';

    // 限制只能从 BASE_DIR 读取，避免路径穿越
    const normalizedPath = path.normalize(path.join(BASE_DIR, '.' + pathname));
    if (!normalizedPath.startsWith(BASE_DIR)) {
        res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Forbidden');
        return;
    }

    const extname = path.extname(normalizedPath);
    const contentType = MIME_TYPES[extname] || 'application/octet-stream';

    fs.readFile(normalizedPath, (error, content) => {
        if (error) {
            if(error.code === 'ENOENT') {
                res.writeHead(404);
                res.end('File not found');
            } else {
                res.writeHead(500);
                res.end('Server error: ' + error.code);
            }
        } else {
            res.writeHead(200, { 
                'Content-Type': contentType,
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0'
            });
            res.end(content);
        }
    });
});

function printBanner(listenPort) {
    console.log('='.repeat(60));
    console.log(`🚀 前端服务器已启动`);
    console.log('='.repeat(60));
    console.log(`📍 服务器地址: http://0.0.0.0:${listenPort}`);
    console.log(`💻 本机访问:    http://localhost:${listenPort}`);
    console.log(`🌐 局域网访问:  http://<你的IP地址>:${listenPort}`);
    console.log('='.repeat(60));

    const os = require('os');
    const interfaces = os.networkInterfaces();

    console.log('📡 可用的网络接口:');
    Object.keys(interfaces).forEach(ifaceName => {
        interfaces[ifaceName].forEach(iface => {
            if (iface.family === 'IPv4' && !iface.internal) {
                console.log(`   - ${ifaceName}: http://${iface.address}:${listenPort}`);
            }
        });
    });
    console.log('='.repeat(60));
}

function tryListen() {
    // 失败端口的 listen() 仍会注册“listening”回调；若换端口重试成功，旧回调也会触发，必须清掉
    server.removeAllListeners('listening');
    const onListenError = (err) => {
        if (err.code === 'EADDRINUSE' && port < portMax) {
            console.warn(`端口 ${port} 已被占用，尝试 ${port + 1} …`);
            port += 1;
            server.close(() => tryListen());
            return;
        }
        console.error(err);
        process.exit(1);
    };
    server.once('error', onListenError);
    server.listen(port, '0.0.0.0', () => {
        server.removeListener('error', onListenError);
        printBanner(port);
    });
}

tryListen();

