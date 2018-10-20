import React from 'react';
import { StyleSheet, Text, View, TextInput, Image, TouchableOpacity } from 'react-native';

export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      text: ""
    };
  }
  render() {
    return (
      <View style={styles.container}>
        <Image source={{uri: 'https://raw.githubusercontent.com/sonamghosh/hack_harvard_2018/FlaskServer/musicBoiLogo/musicBoiLogo2.png'}}
              style={{width: '90%', height: '50%'}}/>
        <Text>Transform words into music</Text>
        <TextInput
          style={styles.input}
          placeholder="Paste text or type text here!"
          multiline={true}
          onChangeText={(text) => this.setState({text})}
        />
        <TouchableOpacity>
          <View style={styles.submit}>
            <Text>Create</Text>
          </View>
        </TouchableOpacity>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#03A9F4',
    alignItems: 'center',
    justifyContent: 'center',
  },
  input: {
    width: "90%",
    height: '50%',
    borderRadius: 5,
    padding: 5,
    margin: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  submit: {
    flex: 1,
    borderRadius: 5,
    padding: 5,
    margin: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
