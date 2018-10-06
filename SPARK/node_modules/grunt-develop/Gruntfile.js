
/*
 * @name grunt-develop
 * @author Edward Hotchkiss <edward@edwardhotchkiss.com>
 * @url http://github.com/edwardhotchkiss/grunt-develop
 * @description Grunt Task to run a Node.js Server while developing, auto-reloading on change.
 * @copyright Copyright (c) 2013-2014 Edward Hotchkiss
 * @license MIT
 */

'use strict';

module.exports = function(grunt) {

  // grunt config
  grunt.initConfig({
    // grunt-contrib-jshint
    jshint: {
      files: [
        'Gruntfile.js',
        'tasks/*.js',
        'test/**/*.js',
        '<%= nodeunit.tests %>'
      ],
      options: {
        'laxcomma':true,
        'curly': true,
        'eqeqeq': true,
        'immed': true,
        'latedef': true,
        'newcap': true,
        'noarg': true,
        'sub': true,
        'undef': true,
        'boss': true,
        'eqnull': true,
        'node': true
      }
    },
    // grunt-develop
    develop: {
      server: {
        file: 'test/app.js'
      }
    },
    // grunt-contrib-watch
    watch: {
      jslint: {
        files: ['<%= jshint.files %>'],
        tasks: ['jshint'],
        options: {
          interrupt: true
        }
      },
      nodeunit: {
        files: 'test/**/*.js',
        tasks: ['node-unit']
      }
    },
    // grunt-contrib-nodeunit
    nodeunit: {
      tests: ['test/*_test.js']
    }
  });

  // load plugin tasks
  grunt.loadTasks('tasks');

  // load required npm plugins
  grunt.loadNpmTasks('grunt-contrib-jshint');
  grunt.loadNpmTasks('grunt-contrib-nodeunit');

  grunt.registerTask('looptest', 'Reload the app periodically.', function() {
    var callback = this.async();
    grunt.log.writeln('Waiting...');
    setTimeout(function() {
      grunt.task.run('develop', 'looptest');
      callback();
    }, 2000);
  });

  // default = run jslint and all tests
  grunt.registerTask('default', ['jshint','develop','nodeunit']);

};
