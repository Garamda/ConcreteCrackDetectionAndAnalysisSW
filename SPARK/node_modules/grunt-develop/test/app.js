'use strict';

var server = require('http').createServer(function (request, response) {
  response.writeHead(200, { 'Content-Type': 'text/plain' });
  response.end('Hello World\n');
});

server.listen(8000, '127.0.0.1', function() {
  console.log('app.js running at http://127.0.0.1:8000/');
});