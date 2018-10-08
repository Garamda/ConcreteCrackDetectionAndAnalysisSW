<!doctype html>
<html>
	<meta charset="utf-8">
	<head>
		<script src="//code.jquery.com/jquery.min.js"></script>
		<script src="js/jquery-1.8.2.js"></script>
		<style>
			
			#videobox {
				width: 300px;
				height:100px;
				vertical-align: middle;
				font-size: 24px;
				background-color:#ececec;
				box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
			}
			/* Style the Image Used to Trigger the Modal */
			#imgbox {
			    border-radius: 5px;
			    cursor: pointer;
			    transition: 0.3s;
			}

			#imgbox:hover {opacity: 0.7;}

			/* The Modal (background) */
			.modal {
			    display: none; /* Hidden by default */
			    position: fixed; /* Stay in place */
			    z-index: 1; /* Sit on top */
			    padding-top: 100px; /* Location of the box */
			    left: 0;
			    top: 0;
			    width: 100%; /* Full width */
			    height: 100%; /* Full height */
			    overflow: auto; /* Enable scroll if needed */
			    background-color: rgb(0,0,0); /* Fallback color */
			    background-color: rgba(0,0,0,0.9); /* Black w/ opacity */
			}

			/* Modal Content (Image) */
			.modal-content {
			    margin: auto;
			    display: block;
			    width: 80%;
			    max-width: 700px;
			}

			/* Caption of Modal Image (Image Text) - Same Width as the Image */
			#caption {
			    margin: auto;
			    display: block;
			    width: 80%;
			    max-width: 700px;
			    text-align: center;
			    color: #ccc;
			    padding: 10px 0;
			    height: 150px;
			}

			/* Add Animation - Zoom in the Modal */
			.modal-content, #caption { 
			    animation-name: zoom;
			    animation-duration: 0.6s;
			}

			@keyframes zoom {
			    from {transform:scale(0)} 
			    to {transform:scale(1)}
			}

			/* The Close Button */
			.close {
			    position: absolute;
			    top: 15px;
			    right: 35px;
			    color: #f1f1f1;
			    font-size: 40px;
			    font-weight: bold;
			    transition: 0.3s;
			}

			.close:hover,
			.close:focus {
			    color: #bbb;
			    text-decoration: none;
			    cursor: pointer;
			}

			/* 100% Image Width on Smaller Screens */
			@media only screen and (max-width: 700px){
			    .modal-content {
			        width: 100%;
			    }
			}
			.modalDialog {
			    position: fixed;
			    font-family: Arial, Helvetica, sans-serif;
			    top: 0;
			    right: 0;
			    bottom: 0;
			    left: 0;
			   // background: rgba(0, 0, 0, 0.8);
			    z-index: 99999;
			    opacity:0;
			    -webkit-transition: opacity 400ms ease-in;
			    -moz-transition: opacity 400ms ease-in;
			    transition: opacity 400ms ease-in;
			    pointer-events: none;
			}
			.modalDialog:target {
			    opacity:1;
			    pointer-events: auto;
			}
			.modalDialog > div {
			    width: 0px;
			    height: 0px;
			    position: relative;
			    margin: 1% auto;
			    padding: 5px;
			}
			#openModal img  {
			    width: 800px;
			    height:auto; 
			}

			.close {
			    background: #606061;
			    color: #FFFFFF;
			    line-height: 25px;
			    position: absolute;
			    right: -200px;
			    text-align: center;
			    top: -10px;
			    width: 24px;
			    text-decoration: none;
			    font-weight: bold;
			    -webkit-border-radius: 12px;
			    -moz-border-radius: 12px;
			    border-radius: 12px;
			    -moz-box-shadow: 1px 1px 3px #000;
			    -webkit-box-shadow: 1px 1px 3px #000;
			    box-shadow: 1px 1px 3px #000;
			}
			.close:hover {
			    background: #00d9ff;
			}
			.windows8 {
					position: relative;
							width: 78px;
									height:78px;
											margin:auto;
			}

			.windows8 .wBall {
					position: absolute;
							width: 74px;
									height: 74px;
											opacity: 0;
													transform: rotate(225deg);
																-o-transform: rotate(225deg);
																			-ms-transform: rotate(225deg);
																						-webkit-transform: rotate(225deg);
																									-moz-transform: rotate(225deg);
																											animation: orbit 6.96s infinite;
																														-o-animation: orbit 6.96s infinite;
																																	-ms-animation: orbit 6.96s infinite;
																																				-webkit-animation: orbit 6.96s infinite;
																																							-moz-animation: orbit 6.96s infinite;
			}

			.windows8 .wBall .wInnerBall{
				position: absolute;
				width: 10px;
				height: 10px;
				background: rgb(0,0,0);
				left:0px;
				top:0px;
				border-radius: 10px;
			}
			.windows8 #wBall_1 {
				animation-delay: 1.52s;
				-o-animation-delay: 1.52s;
				-ms-animation-delay: 1.52s;
				-webkit-animation-delay: 1.52s;
				-moz-animation-delay: 1.52s;
			}
			.windows8 #wBall_2 {
				animation-delay: 0.3s;
				-o-animation-delay: 0.3s;
				-ms-animation-delay: 0.3s;
				-webkit-animation-delay: 0.3s;
				-moz-animation-delay: 0.3s;
			}
			.windows8 #wBall_3 {
				animation-delay: 0.61s;
				-o-animation-delay: 0.61s;
				-ms-animation-delay: 0.61s;
				-webkit-animation-delay: 0.61s;
				-moz-animation-delay: 0.61s;
			}
			.windows8 #wBall_4 {
				animation-delay: 0.91s;
				-o-animation-delay: 0.91s;
				-ms-animation-delay: 0.91s;
				-webkit-animation-delay: 0.91s;
				-moz-animation-delay: 0.91s;
			}
			.windows8 #wBall_5 {
				animation-delay: 1.22s;
				-o-animation-delay: 1.22s;
				-ms-animation-delay: 1.22s;
				-webkit-animation-delay: 1.22s;
				-moz-animation-delay: 1.22s;
			}
			@keyframes orbit {
				0% {
					opacity: 1;
					z-index:99;
					transform: rotate(180deg);
					animation-timing-function: ease-out;
				}
				7% {
					opacity: 1;
					transform: rotate(300deg);
					animation-timing-function: linear;
					origin:0%;
				}
				30% {
					opacity: 1;
					transform:rotate(410deg);
					animation-timing-function: ease-in-out;
					origin:7%;
				}
				39% {
					opacity: 1;
					transform: rotate(645deg);
					animation-timing-function: linear;
					origin:30%;
				}
				70% {
					opacity: 1;
					transform: rotate(770deg);
					animation-timing-function: ease-out;
					origin:39%;
				}
				75% {	
					opacity: 1;
					transform: rotate(900deg);
					animation-timing-function: ease-out;
					origin:70%;
				}
				76% {
					opacity: 0;
					transform:rotate(900deg);
				}
				100% {
					opacity: 0;
					transform: rotate(900deg);
				}
			}

			@-o-keyframes orbit {
				0% {
					opacity: 1;
					z-index:99;
					-o-transform: rotate(180deg);
					-o-animation-timing-function: ease-out;
				}
				7% {
					opacity: 1;
					-o-transform: rotate(300deg);
					-o-animation-timing-function: linear;
					-o-origin:0%;
				}
				30% {
					opacity: 1;
					-o-transform:rotate(410deg);
					-o-animation-timing-function: ease-in-out;
					-o-origin:7%;
				}
				39% {
					opacity: 1;
					-o-transform: rotate(645deg);
					-o-animation-timing-function: linear;
					-o-origin:30%;
				}
				70% {
					opacity: 1;
					-o-transform: rotate(770deg);
					-o-animation-timing-function: ease-out;
					-o-origin:39%;
				}
				75% {
					opacity: 1;
					-o-transform: rotate(900deg);
					-o-animation-timing-function: ease-out;
					-o-origin:70%;
				}
				76% {
					opacity: 0;
					-o-transform:rotate(900deg);
				}
				100% {
					opacity: 0;
					-o-transform: rotate(900deg);
				}
			}

			@-ms-keyframes orbit {
				0% {
					opacity: 1;
					z-index:99;
					-ms-transform: rotate(180deg);
					-ms-animation-timing-function: ease-out;
				}
				7% {
					opacity: 1;
					-ms-transform: rotate(300deg);
					-ms-animation-timing-function: linear;
					-ms-origin:0%;
				}
				30% {
					opacity: 1;
					-ms-transform:rotate(410deg);
					-ms-animation-timing-function: ease-in-out;
					-ms-origin:7%;
				}
				39% {
					opacity: 1;
					-ms-transform: rotate(645deg);
					-ms-animation-timing-function: linear;
					-ms-origin:30%;
				}
				70% {
					opacity: 1;
					-ms-transform: rotate(770deg);
					-ms-animation-timing-function: ease-out;
					-ms-origin:39%;
				}
				75% {
					opacity: 1;
					-ms-transform: rotate(900deg);
					-ms-animation-timing-function: ease-out;
					-ms-origin:70%;
				}
				76% {
					opacity: 0;
					-ms-transform:rotate(900deg);
				}
				100% {
					opacity: 0;
					-ms-transform: rotate(900deg);
				}
			}
			@-webkit-keyframes orbit {
				0% {
					opacity: 1;
					z-index:99;
					-webkit-transform: rotate(180deg);
					-webkit-animation-timing-function: ease-out;
				}
				7% {
					opacity: 1;
					-webkit-transform: rotate(300deg);
					-webkit-animation-timing-function: linear;
					-webkit-origin:0%;
				}
				30% {
					opacity: 1;
					-webkit-transform:rotate(410deg);
					-webkit-animation-timing-function: ease-in-out;
					-webkit-origin:7%;
				}
				39% {
					opacity: 1;
					-webkit-transform: rotate(645deg);
					-webkit-animation-timing-function: linear;
					-webkit-origin:30%;
				}
				70% {
					opacity: 1;
					-webkit-transform: rotate(770deg);
					-webkit-animation-timing-function: ease-out;
					-webkit-origin:39%;
				}
				75% {													
					opacity: 1;
					-webkit-transform: rotate(900deg);
					-webkit-animation-timing-function: ease-out;
					-webkit-origin:70%;
				}
				76% {
					opacity: 0;
					-webkit-transform:rotate(900deg);
				}
				100% {
					opacity: 0;
					-webkit-transform: rotate(900deg);
				}
			}

			@-moz-keyframes orbit {
				0% {
					opacity: 1;
					z-index:99;
					-moz-transform: rotate(180deg);
					-moz-animation-timing-function: ease-out;
				}
				7% {
					opacity: 1;
					-moz-transform: rotate(300deg);
					-moz-animation-timing-function: linear;
					-moz-origin:0%;
				}
				30% {
					opacity: 1;
					-moz-transform:rotate(410deg);
					-moz-animation-timing-function: ease-in-out;
					-moz-origin:7%;
				}
				39% {
					opacity: 1;
					-moz-transform: rotate(645deg);
					-moz-animation-timing-function: linear;
					-moz-origin:30%;
				}			
				70% {
					opacity: 1;
					-moz-transform: rotate(770deg);
					-moz-animation-timing-function: ease-out;
					-moz-origin:39%;
				}
				75% {
					opacity: 1;
					-moz-transform: rotate(900deg);
					-moz-animation-timing-function: ease-out;
					-moz-origin:70%;
				}
				76% {
					opacity: 0;
					-moz-transform:rotate(900deg);
				}
				100% {
					opacity: 0;
					-moz-transform: rotate(900deg);
				}
			}
		</style>
	</head>
	<body>
		<div style="text-align:center">
			<div class="headline" style="text-
			align:center;margin:0 auto;height:100px;">
				<img src="../images/spark.png" style="float:left;width:100%; height:300px;margin:0 auto;">
			</div>
			<div id="UploadBox">
		        <!--<span id='UploadArea'>
		            <label for="FileBox">Choose A File: </label><input type="file" id="FileBox"><br>
		            <label for="NameBox">Name: </label><input type="text" id="NameBox"><br>
		 
		            <button  type='button' id='UploadButton' class='Button'>Upload</button>
		        </span>-->
		        
		    </div>
			<div id="wrapper" style="text-align:center;float:center;width:100%;height:820px">
				<div id="videoarea" style="margin-left:200px;margin-top:80px;float:left;width:440px;text-align:center;font-size:1.3em;">
					<img id="imgvideo" style="width:458px;height:458px;">
					<div style="overflow:scroll; margin:0 auto; text-align:center; font-size:14px; width:450px; height:250px; border:4px solid;border-color:#ead9d9" id="videos">
						<div style="float:left; margin:0 auto; text-align:left; font-size:14px; width:100px" id="thumbnail">

						</div>
						<div style="float:right; margin:0 auto; margin-top:5px; text-align:center; font-size:14px; width:333px" id="playlist">

						</div>
						
					</div>
					<form action="/upload" method="post" enctype="multipart/form-data">
			        	<input type="file" name="userfile">
			        	<input onClick="window.location.href='./process.swig'" type="submit" value="파일 업로드하기">
					</form>
				</div>				
				<!--<div style="float:right;margin:0 10px;width:45%;">
					<div id="imglist" style="background:#ffffff">
						
					</div>
					<div style="background:#ffffff">
						<p style="line-height:2em;font-size:1.3em;background:#ffffff">[Crack Information]<br/></p>
						<p style="background:#ffffff"><span style="background:#ffffff">Predicted boxes: <span style="background:#ffffff;color:red">xmin: 3.14  ymin: 14.75 xmax: 830.31 ymax: 205.74</p>
						<br/>
					</div>
				</div>-->
				<div id="imagearea" style="overflow:scroll;height:710px;margin-left:80px;margin-top:80px;float:left;width:440px;font-size:1.3em;background-color: #fdf9f9; border:4px solid;border-color:#ead9d9;">
					<!--<div style="overflow:scroll; padding -left:20px; float:left; font-size:14px; width:440px; height:710px;" id="imagelist">
						<div id="imgbox" style="margin-left:10px;float:left;text-align:middle;width:30%;height:160px; position:static"><img id="imgview" src="../images/{{title}}/{{imgList[0]}}" style="width:130px;height:130px;position:static"><div id="imgname" style="width:130px; height:30px background-color: #111000">{{imgList[0]}}</div></div>
					</div>-->
				
				</div>				
				<div id="infoarea" style="overflow:scroll;height:710px;float:left;margin-left:60px;margin-top:80px;width:440px;position:statici;border:4px solid;border-color:#ead9d9;">
					
					<div style="background:#fdf9f9;width:440px;height:710px">
						<p style="line-height:2em;font-size:1.3em;">[Crack Information]<br/></p>
						<div id="info"></div>
					</div>
				</div>
				<div id="processing" style="width:1500px;height:830px;visibility:hidden;background-color:#ffffff;position:absolute;top:320px;left:200px;opacity:0.9;">	
					<div class="windows8" style="top:240px;">
						<div class="wBall" id="wBall_1">
							<div class="wInnerBall"></div>
						</div>
						<div class="wBall" id="wBall_2">
							<div class="wInnerBall"></div>
						</div>
						<div class="wBall" id="wBall_3">
							<div class="wInnerBall"></div>
						</div>
						<div class="wBall" id="wBall_4">
							<div class="wInnerBall"></div>
						</div>
						<div class="wBall" id="wBall_5">
							<div class="wInnerBall"></div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</body>
	<script type="text/javascript">
		//console.log({{videoList}});
		var x = 0;
		var img_count = {{imglistsize}};
		jQuery(function($){
			var frame_list = "{{frameList}}".split(",");
			
			function compareNumbers(a, b) {
				    return a - b;
			}
			function getnameOfFilename(filename) {
				    Filename = filename.split(".");
				        var _fileExt = Filename[0];
					    return _fileExt;
			}
			var listlen = {{framelistsize}};
			var List = [];
			for (var i = 0; i < listlen; i++){
				    List.push(parseInt(getnameOfFilename(frame_list[i])));
			}
			var lists = List.sort(compareNumbers);
			console.log(lists);

			for(var i = 0; i < {{listsize}}; i++){
				var video_list = "{{videoList}}".split(",");
					
				$("#thumbnail").append('<video width="100" height="100" controls><source src="/videos/' + video_list[i] + '" type="video/mp4"></video>');
				$("#playlist").append("<a href='../video/"+ video_list[i].split(".")[0]  + "'><button id='videobox'>" + video_list[i] + "</button></a>");
			}
			var stop = "stop";											 
			for(var i = 0; i < {{framelistsize}}; i++){
				//var frame_list = "{{frameList}}".split(",");
				//frame_list = frame_list.sort();
				console.log(lists[i]);
				$("#imagearea").append('<div id="img_box" class="imgbox" style="box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);margin-left:10px;margin-top:20px;float:left;text-align:middle;width:30%;height:160px; position:static"><img id="imgview" onclick="changeImage('+lists[i]+','+stop+')" src="/images/{{title}}/'+lists[i]+'.jpg" style="width:130px;height:130px;position:static"><div id="imgname" style="width:130px; height:30px background-color: #111000">'+lists[i]/24+'sec'+'</div></div>');
				$("#imagearea").append('<div id="openModal'+(i+1)+'" class="modalDialog" style="margin-top:200px;width: 800px;height:auto;"><div><a href="#close" title="Close" class="close">X</a> <div><img src="/images/{{title}}/'+lists[i]+'.jpg" style="margin-left:200px;width:600px;height:600px;background-color:#000000"></div>');
				var modal = document.getElementById('myModal');

				// Get the image and insert it inside the modal - use its "alt" text as a caption
				var img = document.getElementById('imgbox');
				var modalImg = document.getElementById("img"+(i+1));
				/*img.onclick = function(){
				    modal.style.display = "block";
				    modalImg.src = this.src;
				}*/

				// Get the <span> element that closes the modal
				var span = document.getElementsByClassName("close")[0];

				// When the user clicks on <span> (x), close the modal
				span.onclick = function() { 
				  modal.style.display = "none";
				}
			}
			var infolist = "{{infomation}}".split(",");
			var infolen = infolist.length;
			for(var i = 0; i < infolen; i++){
				if(infolist[i] !==  ''){
					$("#info").append('<p onclick="changeImage('+lists[i]+','+stop+')" style="text-align: left; margin: 10px; background-color:Yellow; color:red">['+lists[i]/24+'sec]'+infolist[i]+'</p>');
			    	}
			}

		});
		//var x = 0;

		var img_list = "{{imgList}}".split(",");
		function compareNumbers(a, b) {
			return a - b;
		}
		function getnameOfFilename(filename) {
			Filename = filename.split(".");
	        var _fileExt = Filename[0];
			return _fileExt;
		}
		var listlen = {{imglistsize}};
		var List = [];
		for (var i = 0; i < listlen; i++){
			List.push(parseInt(getnameOfFilename(img_list[i])));
		}
		var lists = List.sort(compareNumbers);
		console.log(lists);

		function changeImage(start, status){
			var img_list = "{{imgList}}".split(",");
			var img = document.getElementById("imgvideo");
			x = start / 6 - 1;
			img.src = "/images/{{title}}/"+lists[x]+".jpg";

			if(x >= img_count-1){
				//x = 0;
				return;
			}
			console.log(x);
			if(status == 'run'){
				setTimeout("changeImage((x+2) * 6, 'run')", 250);
			}else{
				console.log('sdfsdf');
				img_count = 0;
				return;
			}
		}
		changeImage(6, 'run');
		/*function changeStart(start){
			console.log(start);
			x = start / 6 - 1;
			img_count = x;
			console.log(x);
		}*/
		function process_img(){
			document.getElementById("process_img").style.visibility=hidden;
		}

	</script>
</html>
