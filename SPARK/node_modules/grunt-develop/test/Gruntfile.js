
module.exports = function(grunt) {

  grunt.initConfig({
    develop: {
      server: {
        file: 'app.js'
      }
    },
    watch: {
      js: {
        files: [
          'app.js'
        ],
        tasks: ['default'],
        options: { nospawn: true }
      }
    }
  });

  grunt.task.loadTasks('../tasks/');
  grunt.task.loadTasks('../node_modules/grunt-contrib-watch/tasks/');

  grunt.registerTask('default', ['develop','watch']);

};