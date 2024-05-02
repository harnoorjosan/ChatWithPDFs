css = '''
<style>
.chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex}
.chat-message.user {background-color: #2b313e}
.chat-message.bot {background-color: #475063}
.chat-message .avatar {width:15%;}
.chat-message .avatar img {max-width: 78px; max-height: 78px; border-radius: 50%; object-fit: cover} 
.chat-message .message {width: 85%; padding:0 1.5rem; color: #fff;}
'''

bot_template = '''
<div class ="chat-message bot">
    <div class="avatar">
        <img src ="https://st3.depositphotos.com/8950810/17657/v/450/depositphotos_176577870-stock-illustration-cute-smiling-funny-robot-chat.jpg">
    </div>
    <div class="mesage">{{MSG}}</div>
</div>
'''
user_template = '''
<div class ="chat-message user">
    <div class="avatar">
        <img src ="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTm4n2fSbUxf5lqqAzOUp910ozZJ71-HpBXanaTCofIZQ&s">
    </div>
    <div class="mesage">{{MSG}}</div>
</div>
'''