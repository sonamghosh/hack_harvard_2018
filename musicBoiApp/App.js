import React from 'react';
import { StyleSheet, Text, View, TextInput, Image, TouchableOpacity, KeyboardAvoidingView, StatusBar, ImageBackground } from 'react-native';

export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      text: "Paste text or type text here!",
      test: "lol",
    };
  }

  submitText() {
    this.textInput.clear();
    let text = this.state.text;
    fetch('http://musicbois.serveo.net/submitString', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: '&text='+text
    })
      .then(res => res.json())
      .then((resp) => {
        this.setState({test: resp.text});
      })
  }


  render() {
    let test = this.state.test;
    return (
      <ImageBackground
        source={{uri: 'https://thumbs.gfycat.com/DisguisedTangibleArmyant-size_restricted.gif'}}
        style={{width: '100%', height: '100%', resize: 'center'}}
      >
        <KeyboardAvoidingView style={styles.container} behavior="padding" enabled>
          <StatusBar hidden/>
          <Image source={{uri: 'https://raw.githubusercontent.com/sonamghosh/hack_harvard_2018/FlaskServer/musicBoiLogo/musicBoiLogo5.png'}}
                style={{width: 139, height: 85}}/>
          <Text style={styles.text}>Transform words into music</Text>
          <Text style={styles.text}>{test}</Text>
          <TextInput
            style={styles.input}
            placeholder="Paste text or type text here!"
            multiline={true}
            ref={input => { this.textInput = input }}
            onChangeText={(text) => this.setState({text})}
          />
          <TouchableOpacity
            onPress={this.submitText.bind(this)}
          >
            <View style={styles.submit}>
              <Text style={styles.text}>Create</Text>
            </View>
          </TouchableOpacity>
        </KeyboardAvoidingView>
      </ImageBackground>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
    color: '#87ceeb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: '#87ceeb',
  },
  input: {
    width: "80%",
    height: '20%',
    borderRadius: 2,
    padding: 5,
    margin: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    textAlign: 'center',
    opacity: 0.5,
  },
  submit: {
    width: 200,
    height: 50,
    borderRadius: 2,
    padding: 5,
    margin: 10,
    justifyContent: 'center',
    alignItems: 'center',
    color: '#87ceeb',
    opacity: 0.8,
    borderColor: 'skyblue',
    borderWidth: 1,
  },
});
