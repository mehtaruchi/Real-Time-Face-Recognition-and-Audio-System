var start_identifying = document.getElementById("start_recording")
var recdiv = document.getElementById("recorder")

var isRecording = false;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}



start_identifying.onclick =async function() {
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
        xhr.open("POST", "/identify_status");
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr.send(JSON.stringify({ status: "true" }));
        isRecording=true;
    }
    await sleep(10000);
    // XMLHttpRequest
    recdiv.style.background='black';
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);

            // enable download link
        }
    }
    xhr.open("POST", "/identify_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "false" }));
    isRecording=false;
};