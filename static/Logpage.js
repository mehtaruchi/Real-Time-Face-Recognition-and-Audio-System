var logform = document.getElementById('logform')
var show = document.getElementById('show')

show.onclick = function(){
	console.log('button clicked')
	if(logform.style.display=='none'){
		logform.style.display = 'block';
	}
	else{
		logform.style.display = 'None';
	}
}