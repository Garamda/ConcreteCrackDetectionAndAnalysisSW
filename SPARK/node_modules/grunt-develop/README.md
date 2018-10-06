
# grunt-develop [![Build Status](https://secure.travis-ci.org/edwardhotchkiss/grunt-develop.png)](http://travis-ci.org/edwardhotchkiss/grunt-develop)

> Run a Node.js application for development, with support for auto-reload.

## Notes:

  * Requires Grunt >= 0.4.1 && Node.js >= 0.10.0
  * ~~Does not provide a file-watch, [grunt-contrib-watch](https://github.com/gruntjs/grunt-contrib-watch) is helpful here~~
  * No need to modify/export your server or alter your applications code
  * ~~Non-blocking (the task completes immediately and the application will un in the background);~~ Run as last task
  * ~~Reloads cleanly the application when the task is called again,
    allowing for auto-reload.~~

## Contributing

This project has a lot of active contributors and users. Please submit a test along with any changes made. Thanks!

## Install

```bash
$ npm install grunt-develop
```

## Basic Gruntfile.js Example

```javascript
module.exports = function(grunt) {

  grunt.initConfig({
    develop: {
      server: {
        file: 'app.js',
        nodeArgs: ['--debug'],            // optional
        args: ['appArg1', 'appArg2']      // optional
        env: { NODE_ENV: 'development'}      // optional
      }
    }
  });

  grunt.loadNpmTasks('grunt-develop');

  grunt.registerTask('default', ['develop']);

};
```

## Coffeescript App Example

You may also have develop automatically restart coffeescript based node
applications by using the `cmd` option.  This option allows the user to
specify which command/executable to use when restarting the server.

```coffeescript
module.exports = (grunt) ->

  grunt.initConfig
    develop:
      server:
        file: 'app.coffee'
        cmd: 'coffee'

  grunt.loadNpmTasks 'grunt-develop'

  grunt.registerTask 'default', ['develop']
```

## A more complex Gruntfile.js

 To support auto-reload on changes, for example:

```javascript
module.exports = function(grunt) {

  grunt.initConfig({
    watch: {
      js: {
        files: [
          'app.js',
          'routes/**/*.js',
          'lib/*.js'
        ],
        tasks: ['develop'],
        options: { nospawn: true }
      }
    },
    develop: {
      server: {
        file: 'app.js'
      }
    }
  });

  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-develop');

  grunt.registerTask('default', ['develop']);

};
```

The [nospawn](https://github.com/gruntjs/grunt-contrib-watch/blob/master/README.md#optionsnospawn)
is required to keep the grunt context in which `grunt-develop` is running
your application.

Then you can run grunt as the following and get automatic restart of the application on file changes:

```bash
$ grunt
```

You may add any other task in the watch, like JS linting, asset compiling,
etc. and customize the watch to your needs. See
[grunt-contrib-watch](https://github.com/gruntjs/grunt-contrib-watch).

## License (MIT)

Copyright (c) 2013, Edward Hotchkiss.

## Author: [Edward Hotchkiss][0]

[0]: http://github.com/edwardhotchkiss/

[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/edwardhotchkiss/grunt-develop/trend.png)](https://bitdeli.com/free "Bitdeli Badge")
