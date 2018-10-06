
/*
 * @module grunt-develop
 * @author Edward Hotchkiss <edward@edwardhotchkiss.com>
 * @description http://github.com/edwardhotchkiss/grunt-develop/
 * @license MIT
 */

'use strict';

module.exports = function(grunt) {

  var child
    , running = false
    , fs = require('fs')
    , util = require('util');

  // kills child process (server)
  grunt.event.on('develop.kill', function() {
    grunt.log.warn('kill process');
    child.kill('SIGKILL');
  });

  // spawned, notify grunt to move onto next task
  grunt.event.on('develop.started', function() {
    setTimeout(function() {
      global.gruntDevelopDone();
    }, 250);
  });

  // starts server
  grunt.event.on('develop.start', function(filename, nodeArgs, args, env, cmd) {
    var spawnArgs = nodeArgs.concat([filename], args);
    if (running) {
      return grunt.event.emit('develop.kill');
    }
    child = grunt.util.spawn({
      cmd: cmd,
      args: spawnArgs,
      opts: {
        env: env
      }
    }, function(){});
    // handle exit
    child.on('exit', function(code, signal) {
      running = false;
      if (signal !== null) {
        grunt.log.warn(util.format('application exited with signal %s', signal));
      } else {
        grunt.log.warn(util.format('application exited with code %s', code));
      }
      if (signal === 'SIGKILL') {
        grunt.event.emit('develop.start', filename, nodeArgs, args, env, cmd);
      }
    });
    child.stderr.on('data', function(buffer) {
      if (buffer.toString().trim().length) {
        grunt.log.write('\r\n[grunt-develop] > '.red + buffer.toString());
      }
    });
    child.stdout.on('data', function (buffer) {
      grunt.log.write('\r\n[grunt-develop] > '.cyan + buffer.toString());
    });
    running = true;
    grunt.log.write('\r\n[grunt-develop] > '.cyan + util.format('started application "%s".', filename));
    grunt.event.emit('develop.started');
  });

  // TASK. perform setup
  grunt.registerMultiTask('develop', 'init', function() {
    var filename = this.data.file
      , nodeArgs = this.data.nodeArgs || []
      , args = this.data.args || []
      , env = this.data.env || process.env || {}
      , cmd = this.data.cmd || process.argv[0];
    if (!grunt.file.exists(filename)) {
      grunt.fail.warn(util.format('application file "%s" not found!', filename));
      return false;
    }
    global.gruntDevelopDone = this.async();
    grunt.event.emit('develop.start', filename, nodeArgs, args, env, cmd);
  });

  process.on('exit', function() {
    if (running) {
      child.kill('SIGINT');
    }
  });

};
