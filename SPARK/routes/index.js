(function(){
var childProcess = require("child_process");
var oldSpawn = childProcess.spawn;
function mySpawn(){
        console.log('spawn called');
        console.log(arguments);
var result = oldSpawn.apply(this, arguments);
return result;
}
    childProcess.spawn = mySpawn;
})();
console.log( process.env.PATH );

var express = require('express');
var router = express.Router();
var videoDir = './public/videos';
var imageDir = './assets/image';
var logDir = './public/images'
var fs = require('fs');
const path = require('path');
var multer = require('multer')

let {PythonShell} = require('python-shell');

var storage = multer.diskStorage({
	destination: function (req, file, cb) {
		cb(null, 'public/videos') // cb 콜백함수를 통해 전송된 파일 저장 디렉토리 설정
	},
	filename: function (req, file, cb) {
		cb(null, file.originalname) // cb 콜백함수를 통해 전송된 파일 이름 설정
	}
})
var upload = multer({ storage: storage })

/* GET home page. */
router.get('/', function(req, res, next) {
	console.log("sdfsdfs");
	var videolist = fs.readdirSync(videoDir);
	console.log(videolist);
	//var loglist = fs.readdirSync();
	//videolist = filelist;
	videolist = (videolist.length>0)?videolist:[];
	var init_title = videolist.length>0?videolist[0]:'NO VIDEO';
	//console.log(videolist[0].split('.')[0]);
	if(videolist.length!=0)
		res.redirect('/video/'+videolist[0].split('.')[0]);
	else{
		res.render('./index', {
			title: init_title,
			videoList: videolist,
			listsize: videolist.length
		});
	}
});
// videoname을 이용한 python파일 실행 및 이미지,텍스트 정보 send
router.get('/video/:name', function(req, res){
	//1. videoname으로 videoname 보내주기
	//todo: python-shell을 통한 code 실행 ->각각의 디렉토리에 이미지 텍스트 구현 완료 되었다 가정한 개발
	
	var filename = req.params.name;
	console.log(filename);
	var videolist = fs.readdirSync(videoDir);

	//전체 이미지 이름 리스트
	var imglist = fs.readdirSync('./public/images/'+filename+'/');
	//균열 감지된 이미지 이름 리스트
	var framelist = fs.readdirSync('./public/images/'+filename+'_crack/');
	//균열 감지된 이미지 정보 이름 리스트
	//var textlist = fs.readdirSync('./public/images/'+filename+'_info/');
	// var files = fs.readdirSync('C:\Users\rlaal\Desktop\frame');

	//send 균열정보
	var txtoptions = {encoding:'utf-8', flag:'r'};
	
	
	//width
	var widthbuffer = fs.readFileSync('./public/logs/'+filename+'/width.txt', txtoptions);
	var width = widthbuffer.split("\n")
	console.log(width);
	//gps_x
	var gpsxbuffer = fs.readFileSync('./public/logs/'+filename+'/gps_x.txt', txtoptions);
	var gps_x = gpsxbuffer.split("\n");
	console.log("aaaa"+gps_x);
	//gps_y
	var gpsybuffer = fs.readFileSync('./public/logs/'+filename+'/gps_y.txt', txtoptions);
	var gps_y = gpsybuffer.split("\n");
	console.log("aaaa"+gps_y);
	//risk
	var riskbuffer = fs.readFileSync('./public/logs/'+filename+'/risk.txt', txtoptions);
	var risk = riskbuffer.split("\n");
	console.log(risk);
	console.log(videolist);
	console.log(framelist);
	res.render('./index', {
		title: filename,
		videoList: videolist,
		listsize: videolist.length,
		imgList: imglist,
		imglistsize: imglist.length,
		frameList: framelist,
		framelistsize: framelist.length,
		WidthList: width,
		GpsxList: gps_x,
		GpsyList: gps_y,
		RiskList: risk
	});
  });

/*multiparty를 이용한 파일 업로드*/
router.post('/upload', upload.single('userfile'), function(req, res){
	let options = {
		ode: 'text',
		pythonPath: '',
		pythonOptions: ['-u'], // get print results in real-time
		scriptPath: '',
		args: [req.file.filename.split('.')[0]]
	};
	
	req.setTimeout(0); // no timeout
	PythonShell.run('/usr/local/lib/python3.5/dist-packages/tensorflow/keras/ssd_keras/crack.py', options, function (err, results) {
		if (err) throw err;
		res.status(200);
		res.redirect('/video/'+req.file.filename.split('.')[0]);
	});
});

module.exports = router;
