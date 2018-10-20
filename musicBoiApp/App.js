import React from 'react';
import { StyleSheet, Text, View, TextInput, Image, TouchableOpacity, KeyboardAvoidingView, StatusBar, ImageBackground } from 'react-native';

export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      text: ""
    };
  }

  submitText() {
    var newHP = {'currentHP':hp};
    fetch('http://musicbois.serveo.net/submitString', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(newHP)
    })
      .then(res => res.json())
      .then((resp) => {
      })
  }


  render() {
    return (
      <ImageBackground
        source={{uri: 'https://thumbs.gfycat.com/DisguisedTangibleArmyant-size_restricted.gif'}}
        style={{width: '100%', height: '100%', resize: 'center'}}
      >
        <KeyboardAvoidingView style={styles.container} behavior="padding" enabled>
          <StatusBar hidden/>
          <Image source={{uri: 'https://raw.githubusercontent.com/sonamghosh/hack_harvard_2018/FlaskServer/musicBoiLogo/musicBoiLogo4.png'}}
                style={{width: 139, height: 85}}/>
          <Text>Transform words into music</Text>
          <TextInput
            style={styles.input}
            placeholder="Paste text or type text here!"
            multiline={true}
            onChangeText={(text) => this.setState({text})}
          />
          <TouchableOpacity
            onPress={this.submitText}
          >
            <View style={styles.submit}>
              <Text>Create</Text>
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

    alignItems: 'center',
    justifyContent: 'center',
  },
  input: {
    width: "80%",
    height: '40%',
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
    
    opacity: 0.8,
    borderColor: 'skyblue',
    borderWidth: 0.5,
  },
});
