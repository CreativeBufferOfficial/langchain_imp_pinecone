let interval = 0
$(document).ready(function() {


//       var video = document.getElementById("first_video")
//       video.oncanplaythrough = function() {
//            video.muted = false;
//            video.play();
//        }
  // Code to be executed when the DOM is ready


      // const inputElement = document.getElementById("text_input");
      // inputElement.addEventListener("keyup", function(event) {
      //       if (event.keyCode === 13) {
      //       event.preventDefault();
      //       // Code to be executed when Enter key is pressed
      //       // alert("Enter key pressed!");
      //       const value = inputElement.value.trim();
      //       if (value !== "") {
      //         sendAjaxRequest()
      //       }
      //    }
      // });
      num = 0

//      $(".first_response_div>.background-bg video").prop('muted', false);
      autoplayVideo();

      $(".go_back").click(function(){
//      alert()
      window.history.back();
      });

      $("#text_input").on("keydown",function(event){
        handleKeyPress(event,true)
       });
      $('#button_chat').click(function(event) {
        // Prevent the form from submitting
       const value = $('#text_input').val().trim()
       if (value !== "") {
            sendAjaxRequest()
        }
        event.preventDefault();

        });
});

function sendAjaxRequest(){
var csrfToken = document.getElementsByName("csrfmiddlewaretoken")[0].value;
console.log(csrfToken)
        var data_by_user = $('#text_input').val();
        console.log(data_by_user)
        questionClone = $('.question_div_to_be_append').clone()
        questionClone.removeClass("d-none").removeClass("question_div_to_be_append")
        questionClone.find('p').text(data_by_user)
        $('.chat_message').append(questionClone)
        
//        $('#text_input1').val(data_by_user)
        // Get the form data
        var formData = {
          prompt: $('#text_input').val(),
        };
        $('#text_input').val('');
        $('.chat-avatar').removeClass('d-none')
        // Send a POST Ajax request to the server
        $.ajax({
          type: 'POST',
          headers: {'X-CSRFToken': csrfToken},
          url: '/chatmessage/',
          data: formData,
          beforeSend: function(){
            $('.chat_message').append('<div class="loading_area"><div class="loader-img"><img src="/static/images/loading.gif" /></div></div>')
            $("#button_chat").prop("disabled", true);
            interval = 1
          },
          complete: function(){
            // $('.loading_area').addClass('d-none')
            $('.loading_area').remove();

            $('.chat_message').find('div.loading_area').remove()
            
            // $('<div class="loading_area">Doctor is thinking<div class="loader-img"><img src="/static/images/loading.gif" /></div></div>').hide();

            $("#button_chat").prop("disabled", false);
            interval = 0
          },
          success: function(response) {
            console.log(response)
            data = response["response_by_AI"]
            num = num + 1

            answerClone = $('.answer_div_to_be_append').clone()
            answerClone.removeClass("d-none").removeClass("answer_div_to_be_append")
            answerClone.find('p').text(data)
            $('.chat_message').append(answerClone)
            // <div class="loading_area `+num+`"
//


            $('.chat-avatar').addClass('d-none');
            $(".result_video").html('')
             $(".result_video").append(`<video width="60%" height="100%" autoplay muted loop>
                <source src="/static/result_voice.mp4" type="video/mp4" id="chat_avatar">
              </video>`)
            $(".background-bg").html('')
            $(".background-bg").append(`<video width="60%" height="100%" autoplay muted loop>
              <source src="/static/result_voice.mp4" type="video/mp4" id="chat_avatar">
            </video>`)
             $(".result_video video").prop('muted', false);
             $(".result_video video").prop('loop', false);
             $(".background-bg video").prop('loop', false);
             $(".result_video video").on('ended', function() {
                $('.chat-avatar').removeClass('d-none');
                $(".result_video").html('')


             });


            //            var gifElement = document.getElementById('gif_id');
            //            var videoUrl = 'GynoCall\gynochat\static\result_voice.mp4';

            //            videoElement.src = videoUrl;
            //            videoElement.autoplay = true;
            //            videoElement.controls = true;
            //            gifElement.parentNode.replaceChild(videoElement, gifElement);

            $('.response_message').html(response["response_by_AI"]);

            },
          error: function(){
              answerClone = $('.answer_div_to_be_append').clone()
                answerClone.removeClass("d-none").removeClass("answer_div_to_be_append")
                answerClone.find('p').text("Sorry?")
                $('.chat_message').append(answerClone)
              }
        });
}
// function handleKeyPress(event) {
//    if (event.keyCode === 13) {
//    event.preventDefault();
// }

function handleKeyPress(event,attr) {
  if (attr){
    if (event.keyCode === 13 & interval === 0) {
      event.preventDefault();
      const value = $('#text_input').val().trim();
      if (value !== "") {
        sendAjaxRequest()
      }
  }
  }    
}

// function enableEnterKey(attr) {
//    $("#text_input").on("keydown",function(event){
//     handleKeyPress(event,attr)
//    });
//   }
// function disableEnterKey(attr) {
//   $("#text_input").on("keydown",function(event){
//     handleKeyPress(event,attr)
//    });
//   }


function autoplayVideo(){
$(".first_response_div>.background-bg video").prop('loop', false);
//
      $(".first_response_div>.chat-avatar>video").prop('loop', false);
//      $(".first_response_div>.chat-avatar>video").prop('muted', false);
       $(".first_response_div>.chat-avatar>video").on('ended', function() {
                $('.response_div').removeClass('d-none')
                $('.first_response_div').addClass('d-none')
                $('.response_div>.chat-avatar').removeClass('d-none');
                $(".result_video").html('')

             });}