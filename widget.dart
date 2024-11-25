import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Widgets Demo',
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter Widgets Demo'),
        backgroundColor: Colors.blue,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Using the Text widget
            Text(
              'Welcome to Flutter!',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
              ),
            ),
            SizedBox(height: 20), // Adds some space between widgets

            // Using Row to align 3 Containers horizontally
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // First Container
                Container(
                  color: Colors.amber,
                  padding: EdgeInsets.all(20),
                  child: Text(
                    'Ananth',
                    style: TextStyle(fontSize: 18),
                  ),
                ),
                SizedBox(width: 10), // Adds space between containers

                // Second Container
                Container(
                  color: Colors.green,
                  padding: EdgeInsets.all(20),
                  child: Text(
                    'Bhavna',
                    style: TextStyle(fontSize: 18),
                  ),
                ),
                SizedBox(width: 10), // Adds space between containers

                // Third Container
                Container(
                  color: Colors.blue,
                  padding: EdgeInsets.all(20),
                  child: Text(
                    'Khushi',
                    style: TextStyle(fontSize: 18),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20), // Adds some space after the row
          ],
        ),
      ),
    );
  }
}
