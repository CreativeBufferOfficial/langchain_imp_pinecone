
// multi step form 
$('.next').click(function(){
    $('.inside_form > section.active').next('section').addClass('active');
    $(this).parents('section').removeClass('active');
    $('.form-steps ul li.active').removeClass('active').next('li').addClass('active');
});
$('.prev').click(function(){
    $('.inside_form > section.active').prev('section').addClass('active');
    $(this).parents('section').removeClass('active');
    $('.form-steps ul li.active').removeClass('active').prev('li').addClass('active');
});


// sidebar hide and show
$('.toggle').click(function(){
    $(this).parent('.header').toggleClass('full-header')
});



// modal multiple forms
$(document).ready(function(){
    $("#avatarModal").modal('show');
});

$('.next-modal').click(function(){
    $(this).parents('.modal-section').removeClass('active').next().addClass('active');
});
$('.close-modal').click(function(){
    $(this).parents('.modal-section').removeClass('active')
    setTimeout(function() {
        $('.modal-section:nth-child(1)').addClass('active')
    }, 1000);
});

// add remove card
$('.card-data.add_new').click(function(){
    $('.add_new_card_fields').show();
})
$('.close_payment').click(function(){
    $('.add_new_card_fields').hide();
})