$(function(){
	$('#sendString').click(function(){
		chrome.storage.sync.get('storedString', function(textToSend){
			var stringInput = $('#stringInput').val();
      if (stringInput) {
        var returnedTone = '';
        var returnedLink = '';

        chrome.storage.sync.set({'storedString': stringInput})

        let url = 'https://musicbois.serveo.net/submitString';
        let input = stringInput;

        $('#stringInput').val('');
        $('#tone').text('Calculating...')
        $('#link').text('Generating...')
 
        fetch(url, {
          method: 'POST',
          datatype: "json",
          headers: {"Content-Type": "application/x-www-form-urlencoded"},
          body: "&text="+input
        })
        .then(res => res.json())
        .then((resp) => {
          returnedTone = resp.text;
          returnedLink = resp.link;
          $('#tone').text(returnedTone);
          $('#link').text(returnedLink);
        })
      }
      else {
        $('#tone').text('Please enter a text!')
        $('#link').text('Please enter a text!')
      }
		});
	});
});