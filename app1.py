from flask import Flask, request, jsonify
from pymongo import MongoClient


app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/rag_system')
db = client['rag_system']  # Replace 'your_database_name' with your actual database name
users_collection = db['users']

@app.route('/add_user', methods=['POST','GET'])
def add_user():
    if request.method == 'POST':
        data = request.get_json()

        if 'name' not in data:
            return jsonify({'error': 'Missing name'}), 400

        name = data['name']

        # Check if the user already exists
        existing_user = users_collection.find_one({'name': name})
        if existing_user:
            return jsonify({'error': 'User already exists'}), 400

        # Insert new user into MongoDB
        new_user = {'name': name}  # Adding 'intissar' key-value pair directly
        users_collection.insert_one(new_user)

        return jsonify({'message': 'User added successfully'}), 201
    elif request.method == 'GET':
        # Retrieve all users from MongoDB
        all_users = list(users_collection.find({}, {'_id': 0}))  # Exclude '_id' field from results
        return jsonify(all_users), 200

if __name__ == '__main__':
    app.run(debug=True)