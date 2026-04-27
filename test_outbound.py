"""
Outbound test with agent dispatch:
1) Dispatch ai-voice-agent into a room
2) Place outbound SIP call into the same room
"""

import asyncio
import os
from dotenv import load_dotenv
from livekit import api as livekit_api
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.sip import CreateSIPParticipantRequest, CreateSIPOutboundTrunkRequest, SIPOutboundTrunkInfo

load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_SECRET = os.environ["LIVEKIT_API_SECRET"]

# Vobiz outbound trunk credentials
VOBIZ_SIP_DOMAIN = "65516bfe.sip.vobiz.ai"
VOBIZ_USERNAME = "user"
VOBIZ_PASSWORD = "@12348765"
VOBIZ_DID = "+912250658785"

# Test destination and room
CALL_TO = "+918556854513"
ROOM = "outbound-agent-room"
AGENT_NAME = "ai-voice-agent"

# Reuse existing trunk if you already have one, else leave empty to create
TRUNK_ID = "ST_L92aQo24b33p"


async def ensure_trunk(lk: livekit_api.LiveKitAPI) -> str:
    if TRUNK_ID:
        return TRUNK_ID

    print("[1] Creating outbound trunk...")
    trunk = await lk.sip.create_outbound_trunk(
        CreateSIPOutboundTrunkRequest(
            trunk=SIPOutboundTrunkInfo(
                name="vobiz-outbound-test",
                address=VOBIZ_SIP_DOMAIN,
                auth_username=VOBIZ_USERNAME,
                auth_password=VOBIZ_PASSWORD,
                numbers=[VOBIZ_DID],
            )
        )
    )
    return trunk.sip_trunk_id


async def main():
    lk = livekit_api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_KEY,
        api_secret=LIVEKIT_SECRET,
    )

    try:
        trunk_id = await ensure_trunk(lk)
        print(f"[1] Outbound trunk: {trunk_id}")

        print(f"[2] Dispatching agent '{AGENT_NAME}' to room '{ROOM}'...")
        dispatch = await lk.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=ROOM,
                metadata='{"source":"outbound-test"}',
            )
        )
        print(f"[2] Dispatch created: {dispatch.id}")

        print(f"[3] Calling {CALL_TO}...")
        participant = await lk.sip.create_sip_participant(
            CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=CALL_TO,
                room_name=ROOM,
                participant_name="test-call",
            )
        )
        print(f"[3] Call initiated: {participant.sip_call_id}")
        print("Pick up the phone. Agent greeting should play in ~2-3 seconds.")

    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())