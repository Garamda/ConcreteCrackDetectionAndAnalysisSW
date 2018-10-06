var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var swig = require('swig');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var app = express();
//

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.engine('swig', swig.renderFile);
app.set('view engine', 'swig');
app.set('view cache', false); 
swig.setDefaults({ cache: false });


app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use('/', express.static(path.join(__dirname, 'public'))); // public 폴더 static 라우팅

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
