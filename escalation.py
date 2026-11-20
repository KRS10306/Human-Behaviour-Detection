# escalation.py

# Set to False ONLY if you have real Bandwidth credentials and webhooks set up
MOCK_MODE = True 

def trigger_bandwidth_call(message: str, phone_number: str = "+1234567890"):
    """
    Triggers an automated voice call using the Bandwidth Voice API.
    If MOCK_MODE is True, it simulates the API call and prints the payload.
    """
    
    if MOCK_MODE:
        print("\n" + "="*60)
        print(" 🚨 INITIATING EMERGENCY ESCALATION 🚨")
        print(f" DIALING: {phone_number} (Caregiver / Emergency Service)")
        print(f" VOICE MESSAGE: '{message}'")
        print(" [STATUS]: Call connected. Playing Text-to-Speech message.")
        print("="*60 + "\n")
        return "mock-call-id-12345"

    # =========================================================================
    # REAL BANDWIDTH API IMPLEMENTATION
    # Uncomment and fill in your credentials if you have a Bandwidth account.
    # =========================================================================
    try:
        import bandwidth
        from bandwidth.api import calls_api
        from bandwidth.models.create_call import CreateCall

        # 1. Configure the Bandwidth Client
        configuration = bandwidth.Configuration(
            username='YOUR_BANDWIDTH_USERNAME',
            password='YOUR_BANDWIDTH_PASSWORD'
        )
        
        with bandwidth.ApiClient(configuration) as api_client:
            api_instance = calls_api.CallsApi(api_client)
            account_id = 'YOUR_BANDWIDTH_ACCOUNT_ID'
            
            # 2. Create the Call Payload
            # Note: answer_url must be a public URL that returns BXML (Bandwidth XML)
            create_call = CreateCall(
                from_='+19876543210',     # Your Bandwidth phone number
                to=phone_number,          # The caregiver's phone number
                answer_url='https://your-public-server.com/answer', 
                application_id='YOUR_BANDWIDTH_APP_ID',
                tag='fall-alert'
            )
            
            # 3. Dispatch the Call
            api_response = api_instance.create_call(account_id, create_call)
            print(f"[BANDWIDTH API] Voice Call Initiated Successfully. Call ID: {api_response.call_id}")
            return api_response.call_id
            
    except ImportError:
        print("[ERROR] bandwidth-sdk is not installed. Run: pip install bandwidth-sdk")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to make Bandwidth call: {e}")
        return None