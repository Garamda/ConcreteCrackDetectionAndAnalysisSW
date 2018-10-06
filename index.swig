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
		</style>
	</head>
	<body>
		<div style="text-align:center">
			<div class="headline" style="text-
			align:center;margin:0 auto;height:100px;">
				<img src="../images/spark.png" style="float:left;position:absolute;width:1200px; height:150px;margin:0 auto;">
			</div>
			<div id="UploadBox">
		        <!--<span id='UploadArea'>
		            <label for="FileBox">Choose A File: </label><input type="file" id="FileBox"><br>
		            <label for="NameBox">Name: </label><input type="text" id="NameBox"><br>
		 
		            <button  type='button' id='UploadButton' class='Button'>Upload</button>
		        </span>-->
		        
		    </div>
			<div id="wrapper" style="text-align:center;float:center;width:1800px;height:820px">
				<div id="videoarea" style="margin-left:200px;margin-top:100px;float:left;width:440px;text-align:center;font-size:1.3em">
					<video width="450" height="450" controls>
						<source src="/videos/{{title}}.mp4" type="video/mp4">
					</video>
					<div style="overflow:scroll; margin:0 auto; text-align:center; font-size:14px; width:450px; height:250px" id="videos">
						<div style="float:left; margin:0 auto; text-align:left; font-size:14px; width:100px" id="thumbnail">

						</div>
						<div style="float:right; margin:0 auto; margin-top:5px; text-align:center; font-size:14px; width:333px" id="playlist">

						</div>
						
					</div>
					<form action="upload" method="post" enctype="multipart/form-data">
			        	<input type="file" name="userfile">
			        	<input type="submit" value="파일 업로드하기">
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
				<div id="imagearea" style="overflow:scroll;height:710px;margin-left:80px;margin-top:100px;float:left;width:440px;font-size:1.3em;background-color: #fdf9f9">
					<!--<div style="overflow:scroll; padding -left:20px; float:left; font-size:14px; width:440px; height:710px;" id="imagelist">
						<div id="imgbox" style="margin-left:10px;float:left;text-align:middle;width:30%;height:160px; position:static"><img id="imgview" src="../images/{{title}}/{{imgList[0]}}" style="width:130px;height:130px;position:static"><div id="imgname" style="width:130px; height:30px background-color: #111000">{{imgList[0]}}</div></div>
					</div>-->
					
				</div>				
				<div id="infoarea" style="float:left;margin-left:80px;margin-top:80px; 10px;width:440px;height:710px;position:static">
					
					<div style="background:#fdf9f9;width:440px;height:710px">
						<p style="line-height:2em;font-size:1.3em;">{{title}}[Crack Information]<br/></p>
						<p style="background:#fdf9f9"><span style="background:#fdf9f9">Predicted boxes: <span style="background:#fdf9f9;color:red">xmin: 3.14  ymin: 14.75 xmax: 830.31 ymax: 205.74</p>
						<br/>
					</div>
				</div>
			</div>
		</div>
	</body>
	<script type="text/javascript">
		//console.log({{videoList}});
		jQuery(function($){
			
			for(var i = 0; i < {{listsize}}; i++){
				var video_list = "{{videoList}}".split(",");
				
				console.log("<button id='videobox'>"+ video_list[i] +"</button>");
				$("#thumbnail").append('<video width="100" height="100" controls><source src="/videos/' + video_list[i] + '" type="video/mp4"></video>');
				$("#playlist").append("<a href='../video/"+ video_list[i].split(".")[0]  + "'><button id='videobox'>" + video_list[i] + "</button></a>");
			}
			for(var i = 0; i < {{imglistsize}}; i++){
				var img_list = "{{imgList}}".split(",");
				img_list = img_list.sort();
				$("#imagearea").append('<div id="imgbox" style="box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);margin-left:10px;margin-top:20px;float:left;text-align:middle;width:30%;height:160px; position:static"><a href="#openModal'+(i+1)+'"><img id="imgview" src="/images/{{title}}/'+(i + 1)+'.jpg" style="width:130px;height:130px;position:static"></a><div id="imgname" style="width:130px; height:30px background-color: #111000">'+(i+1)+'</div></div>');
				$("#imagearea").append('<div id="openModal'+(i+1)+'" class="modalDialog" style="margin-top:200px;width: 800px;height:auto;"><div><a href="#close" title="Close" class="close">X</a> <div><img src="/images/{{title}}/'+(i + 1)+'.jpg" style="margin-left:200px;width:600px;height:600px;background-color:#000000"></div>');
				var modal = document.getElementById('myModal');

				// Get the image and insert it inside the modal - use its "alt" text as a caption
				var img = document.getElementById('imgbox');
				var modalImg = document.getElementById("img"+(i+1));
				img.onclick = function(){
				    modal.style.display = "block";
				    modalImg.src = this.src;
				}

				// Get the <span> element that closes the modal
				var span = document.getElementsByClassName("close")[0];

				// When the user clicks on <span> (x), close the modal
				span.onclick = function() { 
				  modal.style.display = "none";
				}

			}
		});


	</script>
</html>