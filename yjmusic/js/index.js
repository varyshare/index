'use strict';

var audioPlayer = document.querySelector('.green-audio-player');
var playPause = audioPlayer.querySelector('#playPause');
var playpauseBtn = audioPlayer.querySelector('.play-pause-btn');
var loading = audioPlayer.querySelector('.loading');
var progress = audioPlayer.querySelector('.progress');
var sliders = audioPlayer.querySelectorAll('.slider');

var player = audioPlayer.querySelector('audio');
var currentTime = audioPlayer.querySelector('.current-time');
var totalTime = audioPlayer.querySelector('.total-time');
var speaker = audioPlayer.querySelector('#speaker');

var draggableClasses = ['pin'];
var currentlyDragged = null;

window.addEventListener('mousedown', function (event) {

  if (!isDraggable(event.target)) return false;

  currentlyDragged = event.target;
  var handleMethod = currentlyDragged.dataset.method;

  this.addEventListener('mousemove', window[handleMethod], false);

  window.addEventListener('mouseup', function () {
    currentlyDragged = false;
    window.removeEventListener('mousemove', window[handleMethod], false);
  }, false);
});

playpauseBtn.addEventListener('click', togglePlay);
player.addEventListener('timeupdate', updateProgress);

player.addEventListener('loadedmetadata', function () {
  totalTime.textContent = formatTime(player.duration);
});
player.addEventListener('canplay', makePlay);
player.addEventListener('ended', function () {
  playPause.attributes.d.value = "M18 12L0 24V0";
  player.currentTime = 0;
});



window.addEventListener('resize', directionAware);

sliders.forEach(function (slider) {
  var pin = slider.querySelector('.pin');
  slider.addEventListener('click', window[pin.dataset.method]);
});

directionAware();

function isDraggable(el) {
  var canDrag = false;
  var classes = Array.from(el.classList);
  draggableClasses.forEach(function (draggable) {
    if (classes.indexOf(draggable) !== -1) canDrag = true;
  });
  return canDrag;
}

function inRange(event) {
  var rangeBox = getRangeBox(event);
  var rect = rangeBox.getBoundingClientRect();
  var direction = rangeBox.dataset.direction;
  if (direction == 'horizontal') {
    var min = rangeBox.offsetLeft;
    var max = min + rangeBox.offsetWidth;
    if (event.clientX < min || event.clientX > max) return false;
  } else {
    var min = rect.top;
    var max = min + rangeBox.offsetHeight;
    if (event.clientY < min || event.clientY > max) return false;
  }
  return true;
}

function updateProgress() {
  var current = player.currentTime;
  var percent = current / player.duration * 100;
  progress.style.width = percent + '%';

  currentTime.textContent = formatTime(current);
}



function getRangeBox(event) {
  var rangeBox = event.target;
  var el = currentlyDragged;
  if (event.type == 'click' && isDraggable(event.target)) {
    rangeBox = event.target.parentElement.parentElement;
  }
  if (event.type == 'mousemove') {
    rangeBox = el.parentElement.parentElement;
  }
  return rangeBox;
}

function getCoefficient(event) {
  var slider = getRangeBox(event);
  var rect = slider.getBoundingClientRect();
  var K = 0;
  if (slider.dataset.direction == 'horizontal') {

    var offsetX = event.clientX - slider.offsetLeft;
    var width = slider.clientWidth;
    K = offsetX / width;
  } else if (slider.dataset.direction == 'vertical') {

    var height = slider.clientHeight;
    var offsetY = event.clientY - rect.top;
    K = 1 - offsetY / height;
  }
  return K;
}

function rewind(event) {
  if (inRange(event)) {
    player.currentTime = player.duration * getCoefficient(event);
  }
}


function formatTime(time) {
  var min = Math.floor(time / 60);
  var sec = Math.floor(time % 60);
  return min + ':' + (sec < 10 ? '0' + sec : sec);
}

function togglePlay() {
  if (player.paused) {
    playPause.attributes.d.value = "M0 0h6v24H0zM12 0h6v24h-6z";
    player.play();
  } else {
    playPause.attributes.d.value = "M18 12L0 24V0";
    player.pause();
  }
}

function makePlay() {
  playpauseBtn.style.display = 'block';
  loading.style.display = 'none';
}

function directionAware() {
  
}

function msToTime(ms){
    var s = parseInt(ms/1000);
    var m = parseInt(s/60);
    s = parseInt(s%60);
    var h = parseInt(m/60);
    m = parseInt(m%60);
    return h+":"+m+":"+s;
}
var timeID;
function setalarm(){
  playPause.attributes.d.value = "M18 12L0 24V0";
  player.pause();
  
  clearInterval(timeID);
  document.getElementById("state").innerHTML = "无闹钟计划";
  var digit = ['hour', 'minute', 'second'];
  for(var item in digit){
    console.log(digit[item]);
    var val = document.getElementById(digit[item]).value;
    console.log(val.length);
    if(val.length==0)
      document.getElementById(digit[item]).value=(0).toString();
  }
  var hour=Math.abs(parseInt(document.getElementById("hour").value));
  var minute=Math.abs(parseInt(document.getElementById("minute").value));
  var second =Math.abs(parseInt(document.getElementById("second").value));
  if(hour==NaN)
    hour = 0;
  if(minute==NaN)
    minute = 0;
  if(second==NaN)
    second = 0;
    
  var ms = ((hour*60+minute)*60+second)*1000;

  document.getElementById("state").innerHTML = "设置成功";
  timeID = setInterval(function(){document.getElementById("state").innerHTML="时间还剩"+msToTime(ms)+"播放音乐";ms-=1000;},"1000");
  setTimeout(function(){
          if (player.paused) {
            playPause.attributes.d.value = "M0 0h6v24H0zM12 0h6v24h-6z";
            player.play();
          }
          clearInterval(timeID);
          document.getElementById("state").innerHTML = "无闹钟计划";

          }, (ms).toString() );

}