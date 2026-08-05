const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

// 静态网页
app.use(express.static('./public'));

// 代理 5001
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:5001',
  changeOrigin: true,
}));

app.listen(3000, '0.0.0.0', () => {
  console.log('Server running on port 3000');
});
