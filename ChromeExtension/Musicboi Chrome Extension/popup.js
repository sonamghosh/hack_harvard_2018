$(function(){
	$('#sendString').click(function(){
		chrome.storage.sync.get('storedString', function(textToSend){
			var stringInput = $('#stringInput').val();
			var returned = '';

			chrome.storage.sync.set({'storedString': stringInput})

			let url = 'https://musicbois.serveo.net/submitString';
  			let input = stringInput;

  			$('#stringInput').val('');
  			$('#link').text('Generating...')
 
  			fetch(url, {
    			method: 'POST',
    			datatype: "json",
    			headers: {"Content-Type": "application/x-www-form-urlencoded"},
    			body: "&text="+input
    		})
    		.then(res => res.json())
    		.then((resp) => {
    			returned = resp.text;
    			$('#link').text(returned);
    		})
		});
	});
});