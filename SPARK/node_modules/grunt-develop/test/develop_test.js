'use strict';

var grunt = require('grunt');

exports.develop = {

  support: function(test) {
    test.expect(2);
    test.ok(!grunt.file.exists('app.js'), 'app.js fixture should exist');
    test.ok(!grunt.file.exists('app.coffee'), 'app.coffee fixture should exist');
    test.done();
  },

  fail: function(test) {
    test.expect(1);
    grunt.util.spawn({
      grunt: true
    }, function(error, result) {
      test.ok(result.stdout.indexOf('app.js running at http://127.0.0.1:8000/') === -1, 'app.js running at http://127.0.0.1:8000/');
      test.done();
    });
  }

};
