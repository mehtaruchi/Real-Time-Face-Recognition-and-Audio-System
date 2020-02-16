
var buttonRecord = document.querySelector("#recordbut");
var sub = document.getElementById("sub")
var recdiv = document.getElementById("recorder")


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

sub.disabled=true;

var isRecording = false;

buttonRecord.onclick =async function() {
    // var url = window.location.href + "record_status";
    if(isRecording==false){


        // XMLHttpRequest
        recdiv.style.background='red';
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4 && xhr.status == 200) {
                // alert(xhr.responseText);
            }
        }
        xhr.open("POST", "/record_status");
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr.send(JSON.stringify({ status: "true" }));
        isRecording=true;
    }
    await sleep(6000);
    sub.disabled = false;
    // XMLHttpRequest
    recdiv.style.background='black';
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);

            // enable download link
        }
    }
    xhr.open("POST", "/record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "false" }));
    isRecording=false;
};




// sub.onclick = function(){
//     var object = new ActiveXObject("Scripting.FileSystemObject")
//     var counter =1 ;
//     while(counter<=19){
//         var file = object.GetFile("image"+counter+".png");
//         counter+=1;
//         file.Move('../');
//     }
// }