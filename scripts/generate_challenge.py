#!/usr/bin/env python3
"""Generate a CHALLENGE SET of hand-crafted novel scenarios for AURA-Lite.

These scenarios test generalization beyond training templates. Every scenario
uses different wording, structure, or complexity compared to the training data
in generate_scenarios.py.

Challenge categories:
    1. Paraphrased messages (5) — same intent, completely different wording
    2. Multi-issue messages (4) — customer has TWO problems
    3. Angry/frustrated tone (3) — emotionally charged messages
    4. Already-tried steps (3) — customer mentions prior troubleshooting
    5. Contradictory info (3) — inconsistent details in the message
    6. Enough info — should NOT ask (4) — all info present, model should act
    7. Edge cases (3-4) — unknown device, very short, unrelated request

Usage:
    python scripts/generate_challenge.py
"""

import json
import os

# KB topics reused from generate_scenarios.py for consistency
KB_TOPICS = {
    "black_screen": [
        "force_restart_iphone", "force_restart_android",
        "screen_replacement_claim", "water_damage_check",
        "battery_calibration", "dfu_mode_recovery",
    ],
    "charging": [
        "charging_port_cleaning", "cable_troubleshoot",
        "wireless_charging_setup", "battery_replacement_claim",
        "power_adapter_check", "charging_port_repair",
    ],
    "battery_drain": [
        "battery_health_check", "background_app_refresh",
        "location_services_audit", "battery_replacement_claim",
        "low_power_mode", "software_update_check",
    ],
    "activation": [
        "sim_activation_steps", "esim_transfer",
        "carrier_activation_portal", "imei_check",
        "activation_lock_removal", "factory_reset_activation",
    ],
    "wifi_network": [
        "network_reset_steps", "router_troubleshoot",
        "dns_settings", "wifi_calling_setup",
        "airplane_mode_toggle", "forget_network_reconnect",
    ],
    "data_transfer": [
        "icloud_backup_restore", "google_backup_transfer",
        "smart_switch_samsung", "move_to_ios",
        "manual_transfer_usb", "contact_sync_check",
    ],
    "damaged_screen": [
        "screen_replacement_claim", "screen_protector_check",
        "display_calibration", "touch_sensitivity_settings",
        "insurance_claim_eligibility", "authorized_repair_locations",
    ],
    "lost_stolen": [
        "find_my_device", "remote_lock_wipe",
        "police_report_filing", "insurance_claim_lost",
        "imei_blacklist", "replacement_device_claim",
    ],
    "liquid_damage": [
        "liquid_damage_first_aid", "rice_myth_debunk",
        "insurance_claim_liquid", "water_damage_indicator",
        "drying_procedure", "authorized_repair_assessment",
    ],
    "claim_routing": [
        "claim_eligibility_check", "deductible_lookup",
        "claim_status_tracking", "replacement_options",
        "repair_vs_replace", "coverage_verification",
    ],
}


def build_challenge_scenarios():
    """Build all challenge scenarios by hand.

    Returns a list of scenario dicts matching the Scenario.from_dict format:
        - scenario_id
        - state: {customer_message, device_type, issue_category,
                  information_collected, conversation_turn, available_kb_topics}
        - hidden_ground_truth: {true_intent, true_issue, required_information,
                                ideal_action, ideal_clarification,
                                acceptable_actions, acceptable_clarification_topics,
                                resolution_path, should_not_do}
    """
    scenarios = []
    counter = 0

    def add(
        category,
        customer_message,
        device_type,
        issue_category,
        information_collected,
        conversation_turn,
        kb_domain,
        true_intent,
        true_issue,
        required_information,
        ideal_action,
        ideal_clarification,
        acceptable_actions,
        acceptable_clarification_topics,
        resolution_path,
        should_not_do,
    ):
        nonlocal counter
        counter += 1
        scenarios.append({
            "scenario_id": f"challenge_{counter:03d}",
            "challenge_category": category,
            "state": {
                "customer_message": customer_message,
                "device_type": device_type,
                "issue_category": issue_category,
                "information_collected": information_collected,
                "conversation_turn": conversation_turn,
                "available_kb_topics": KB_TOPICS.get(
                    kb_domain, ["general_troubleshooting"]
                )[:4],
            },
            "hidden_ground_truth": {
                "true_intent": true_intent,
                "true_issue": true_issue,
                "required_information": required_information,
                "ideal_action": ideal_action,
                "ideal_clarification": ideal_clarification,
                "acceptable_actions": acceptable_actions,
                "acceptable_clarification_topics":
                    acceptable_clarification_topics,
                "resolution_path": resolution_path,
                "should_not_do": should_not_do,
            },
        })

    # ================================================================
    # CATEGORY 1: PARAPHRASED MESSAGES (5 scenarios)
    # Same intent as training templates, completely different wording
    # ================================================================

    add(
        category="paraphrased",
        customer_message=(
            "Display not showing anything, just dark. Happened out of "
            "nowhere while I was texting. No drops or anything like that."
        ),
        device_type="Samsung Galaxy S24",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="black_screen",
        true_intent="black_screen",
        true_issue="software_crash",
        required_information=[],
        ideal_action="SEARCH_KB",
        ideal_clarification="",
        acceptable_actions=["SEARCH_KB", "PROVIDE_STEP"],
        acceptable_clarification_topics=[],
        resolution_path="force_restart",
        should_not_do=["COMPLETE", "ROUTE_CLAIM"],
    )

    add(
        category="paraphrased",
        customer_message=(
            "The power runs out on my handset way too quickly. I unplug "
            "it in the morning and by lunchtime it is completely dead."
        ),
        device_type="Google Pixel 8",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="battery_drain",
        true_intent="battery_drain",
        true_issue="excessive_drain",
        required_information=[
            "battery_health_percent", "usage_pattern", "device_age",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification="Ask about battery health percentage and usage",
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "battery_health", "usage_pattern", "device_age",
        ],
        resolution_path="battery_diagnostics",
        should_not_do=["COMPLETE", "ROUTE_CLAIM"],
    )

    add(
        category="paraphrased",
        customer_message=(
            "Can't get any juice into the handset. Plugged it in with two "
            "separate cords and neither one did anything. The socket on "
            "the bottom looks clean enough."
        ),
        device_type="iPhone 15",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="charging",
        true_intent="charging",
        true_issue="charging_port_or_software",
        required_information=[],
        ideal_action="SEARCH_KB",
        ideal_clarification="",
        acceptable_actions=["SEARCH_KB", "PROVIDE_STEP"],
        acceptable_clarification_topics=["port_debris", "cable_type"],
        resolution_path="port_cleaning_or_restart",
        should_not_do=["COMPLETE", "ESCALATE"],
    )

    add(
        category="paraphrased",
        customer_message=(
            "The glass on the front is all smashed up. Fell right out of "
            "my pocket onto the sidewalk. Touch still kinda works. Got "
            "the protection plan."
        ),
        device_type="iPhone 16",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="cracked_screen",
        required_information=[],
        ideal_action="ROUTE_CLAIM",
        ideal_clarification="",
        acceptable_actions=["ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="screen_replacement_claim",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="paraphrased",
        customer_message=(
            "Wireless internet keeps cutting in and out on my tablet. "
            "Every other gadget in the house is fine. Tried toggling "
            "airplane mode already."
        ),
        device_type="iPad Air",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="wifi_network",
        true_intent="wifi_network",
        true_issue="wifi_connection_failure",
        required_information=[],
        ideal_action="PROVIDE_STEP",
        ideal_clarification="",
        acceptable_actions=["PROVIDE_STEP", "SEARCH_KB"],
        acceptable_clarification_topics=["error_message", "router_status"],
        resolution_path="network_reset",
        should_not_do=["ROUTE_CLAIM", "ESCALATE"],
    )

    # ================================================================
    # CATEGORY 2: MULTI-ISSUE MESSAGES (4 scenarios)
    # Customer describes TWO distinct problems
    # ================================================================

    add(
        category="multi_issue",
        customer_message=(
            "My screen is cracked AND it won't charge anymore. I dropped "
            "it yesterday and now both things are broken. I have insurance."
        ),
        device_type="Samsung Galaxy S23",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="physical_damage_multiple",
        required_information=[],
        ideal_action="ROUTE_CLAIM",
        ideal_clarification="",
        acceptable_actions=["ROUTE_CLAIM", "ASK_CLARIFICATION"],
        acceptable_clarification_topics=["extent_of_damage"],
        resolution_path="comprehensive_damage_claim",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="multi_issue",
        customer_message=(
            "My phone won't connect to Wi-Fi and the battery is also "
            "draining super fast. Both started after the last update. "
            "Google Pixel 8."
        ),
        device_type="Google Pixel 8",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="wifi_network",
        true_intent="wifi_network",
        true_issue="post_update_multi_issue",
        required_information=[
            "battery_health_percent", "wifi_error_message",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Clarify which issue is primary and get details on each"
        ),
        acceptable_actions=["ASK_CLARIFICATION", "SEARCH_KB"],
        acceptable_clarification_topics=[
            "battery_health", "wifi_error", "update_version",
        ],
        resolution_path="software_update_rollback",
        should_not_do=["COMPLETE", "ROUTE_CLAIM"],
    )

    add(
        category="multi_issue",
        customer_message=(
            "I lost my phone yesterday and I also need to transfer my "
            "data to a new device. The old one had all my photos."
        ),
        device_type=None,
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="lost_stolen",
        true_intent="lost_stolen",
        true_issue="lost_device_data_concern",
        required_information=[
            "lost_or_stolen", "find_my_attempted",
            "backup_status", "device_model",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Determine if device is lost/stolen and if backups exist"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "lost_or_stolen", "backup_availability", "device_model",
        ],
        resolution_path="locate_then_transfer",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="multi_issue",
        customer_message=(
            "Screen has a crack at the top and there are also green "
            "lines flickering across the display. Not sure if the lines "
            "are from the crack or a separate problem. iPhone 15 Pro."
        ),
        device_type="iPhone 15 Pro",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="crack_plus_display_defect",
        required_information=["drop_details", "when_lines_started"],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Ask when lines appeared relative to crack to determine if "
            "it is one issue or two"
        ),
        acceptable_actions=["ASK_CLARIFICATION", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[
            "timeline", "drop_details", "insurance_status",
        ],
        resolution_path="damage_assessment",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    # ================================================================
    # CATEGORY 3: ANGRY/FRUSTRATED TONE (3 scenarios)
    # Emotionally charged messages
    # ================================================================

    add(
        category="angry_frustrated",
        customer_message=(
            "I'm SO frustrated! I've been on hold for an hour and "
            "NOTHING works! My phone is completely dead. This is "
            "ridiculous! I pay for insurance every month and nobody "
            "can help me?!"
        ),
        device_type=None,
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="claim_routing",
        true_intent="claim_routing",
        true_issue="claim_eligibility_check",
        required_information=[
            "damage_type", "device_model", "device_condition_details",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Acknowledge frustration, then ask what happened to the device"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "damage_description", "device_model", "what_happened",
        ],
        resolution_path="eligibility_assessment",
        should_not_do=["COMPLETE", "PROVIDE_STEP", "ESCALATE"],
    )

    add(
        category="angry_frustrated",
        customer_message=(
            "THIS IS THE THIRD TIME I'M CALLING. My Samsung Galaxy S24 "
            "still won't charge after your so-called experts told me to "
            "clean the port and try a new cable. NEITHER worked. I want "
            "this FIXED or give me a new phone!"
        ),
        device_type="Samsung Galaxy S24",
        issue_category=None,
        information_collected=[
            "charging_port_cleaned", "new_cable_tried",
        ],
        conversation_turn=3,
        kb_domain="charging",
        true_intent="charging",
        true_issue="repeated_contact_unresolved",
        required_information=[],
        ideal_action="ESCALATE",
        ideal_clarification="",
        acceptable_actions=["ESCALATE"],
        acceptable_clarification_topics=[],
        resolution_path="supervisor_escalation",
        should_not_do=["PROVIDE_STEP", "COMPLETE", "SEARCH_KB"],
    )

    add(
        category="angry_frustrated",
        customer_message=(
            "Absolutely unbelievable. My brand new iPhone 16 that I got "
            "TWO DAYS AGO already has a dead screen. I didn't drop it, "
            "I didn't get it wet. It just died. I want a replacement NOW."
        ),
        device_type="iPhone 16",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="black_screen",
        true_intent="black_screen",
        true_issue="defective_unit",
        required_information=[],
        ideal_action="SEARCH_KB",
        ideal_clarification="",
        acceptable_actions=["SEARCH_KB", "PROVIDE_STEP", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="force_restart_then_claim",
        should_not_do=["COMPLETE"],
    )

    # ================================================================
    # CATEGORY 4: ALREADY-TRIED STEPS (3 scenarios)
    # Customer mentions prior troubleshooting
    # ================================================================

    add(
        category="already_tried",
        customer_message=(
            "I already restarted three times and reset network settings "
            "but my iPhone 14 still can't connect to Wi-Fi. Other "
            "devices at home work fine. I'm out of ideas."
        ),
        device_type="iPhone 14",
        issue_category=None,
        information_collected=[
            "restart_attempted", "network_settings_reset",
        ],
        conversation_turn=0,
        kb_domain="wifi_network",
        true_intent="wifi_network",
        true_issue="wifi_persistent_failure",
        required_information=[],
        ideal_action="SEARCH_KB",
        ideal_clarification="",
        acceptable_actions=["SEARCH_KB", "PROVIDE_STEP"],
        acceptable_clarification_topics=["error_message"],
        resolution_path="forget_network_reconnect",
        should_not_do=["COMPLETE", "ROUTE_CLAIM"],
    )

    add(
        category="already_tried",
        customer_message=(
            "Battery drains super fast on my Pixel 8. Already turned off "
            "background app refresh, disabled location services for most "
            "apps, and even factory reset the phone. Still dies by 2pm. "
            "Battery health is 91%."
        ),
        device_type="Google Pixel 8",
        issue_category=None,
        information_collected=[
            "background_refresh_disabled", "location_services_audited",
            "factory_reset_done",
        ],
        conversation_turn=0,
        kb_domain="battery_drain",
        true_intent="battery_drain",
        true_issue="persistent_drain_after_troubleshooting",
        required_information=[],
        ideal_action="ESCALATE",
        ideal_clarification="",
        acceptable_actions=["ESCALATE", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="hardware_diagnostic_escalation",
        should_not_do=["PROVIDE_STEP", "COMPLETE"],
    )

    add(
        category="already_tried",
        customer_message=(
            "My Galaxy Tab S9 screen went black. I've done the force "
            "restart (volume up + power), tried DFU mode, and even "
            "plugged it into my computer. Nothing. Screen stays black. "
            "No physical damage."
        ),
        device_type="Samsung Galaxy Tab S9",
        issue_category=None,
        information_collected=[
            "force_restart_attempted", "dfu_mode_attempted",
            "usb_connection_attempted",
        ],
        conversation_turn=0,
        kb_domain="black_screen",
        true_intent="black_screen",
        true_issue="hardware_failure_likely",
        required_information=[],
        ideal_action="ESCALATE",
        ideal_clarification="",
        acceptable_actions=["ESCALATE", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="hardware_escalation",
        should_not_do=["PROVIDE_STEP", "COMPLETE"],
    )

    # ================================================================
    # CATEGORY 5: CONTRADICTORY INFO (3 scenarios)
    # Customer provides inconsistent or conflicting details
    # ================================================================

    add(
        category="contradictory",
        customer_message=(
            "My screen is black but I can see the crack on it. The "
            "phone rings when people call but the display shows nothing. "
            "Actually wait, I can see a tiny bit of light in the corner."
        ),
        device_type=None,
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="screen_damage_unclear",
        required_information=[
            "crack_details", "screen_completely_black_or_partial",
            "drop_history",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Ask if screen is fully black or partially visible, and "
            "how the crack occurred"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "screen_state", "visible_damage", "drop_history",
        ],
        resolution_path="damage_assessment",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="contradictory",
        customer_message=(
            "My phone got soaked in rain yesterday but I don't think any "
            "water got inside. The speaker sounds muffled though and "
            "charging is intermittent. Samsung Galaxy A54."
        ),
        device_type="Samsung Galaxy A54",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="liquid_damage",
        true_intent="liquid_damage",
        true_issue="liquid_damage_denial",
        required_information=[
            "exposure_duration", "current_symptoms", "insurance_status",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Clarify extent of water exposure since symptoms suggest "
            "internal damage despite claim of no water inside"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "exposure_details", "symptoms", "insurance",
        ],
        resolution_path="liquid_damage_assessment",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="contradictory",
        customer_message=(
            "I never dropped my phone but the screen is cracked. I "
            "literally just pulled it out of my bag and it was like "
            "this. Maybe I sat on it? I'm not sure. Motorola Edge."
        ),
        device_type="Motorola Edge",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="unexplained_crack",
        required_information=[
            "how_discovered", "bag_contents", "insurance_status",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Clarify circumstances and check insurance for claim routing"
        ),
        acceptable_actions=["ASK_CLARIFICATION", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[
            "circumstances", "insurance_status", "visible_damage",
        ],
        resolution_path="damage_assessment",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    # ================================================================
    # CATEGORY 6: ENOUGH INFO -- SHOULD NOT ASK (4 scenarios)
    # All required info is present; model should act, not clarify
    # ================================================================

    add(
        category="enough_info",
        customer_message=(
            "My iPhone 15 Pro screen is shattered from a drop on "
            "concrete. Touch doesn't work at all. I have active "
            "insurance through Asurion. I'd like to file a claim for "
            "screen replacement."
        ),
        device_type="iPhone 15 Pro",
        issue_category="damaged_screen",
        information_collected=[
            "damage_type_confirmed", "insurance_verified",
            "device_model_confirmed",
        ],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="cracked_screen",
        required_information=[],
        ideal_action="ROUTE_CLAIM",
        ideal_clarification="",
        acceptable_actions=["ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="screen_replacement_claim",
        should_not_do=["ASK_CLARIFICATION", "COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="enough_info",
        customer_message=(
            "Galaxy S23 won't connect to my home Wi-Fi. Password is "
            "correct. Other phones connect fine. I already toggled "
            "airplane mode. What's the next step?"
        ),
        device_type="Samsung Galaxy S23",
        issue_category="wifi_network",
        information_collected=["airplane_mode_toggled"],
        conversation_turn=0,
        kb_domain="wifi_network",
        true_intent="wifi_network",
        true_issue="wifi_connection_failure",
        required_information=[],
        ideal_action="PROVIDE_STEP",
        ideal_clarification="",
        acceptable_actions=["PROVIDE_STEP", "SEARCH_KB"],
        acceptable_clarification_topics=[],
        resolution_path="forget_network_reconnect",
        should_not_do=["ASK_CLARIFICATION", "ROUTE_CLAIM", "COMPLETE"],
    )

    add(
        category="enough_info",
        customer_message=(
            "Dropped my iPad Pro in the bathtub. It was fully submerged "
            "for about 30 seconds. Won't power on at all now. I have "
            "insurance. Please start the claim."
        ),
        device_type="iPad Pro",
        issue_category="liquid_damage",
        information_collected=[
            "exposure_type_confirmed", "insurance_verified",
        ],
        conversation_turn=0,
        kb_domain="liquid_damage",
        true_intent="liquid_damage",
        true_issue="liquid_damage_total",
        required_information=[],
        ideal_action="ROUTE_CLAIM",
        ideal_clarification="",
        acceptable_actions=["ROUTE_CLAIM"],
        acceptable_clarification_topics=[],
        resolution_path="liquid_damage_claim",
        should_not_do=["ASK_CLARIFICATION", "PROVIDE_STEP", "COMPLETE"],
    )

    add(
        category="enough_info",
        customer_message=(
            "Following up: the force restart you suggested fixed my "
            "Galaxy S24 black screen. Everything is working perfectly "
            "now. Thank you!"
        ),
        device_type="Samsung Galaxy S24",
        issue_category="black_screen",
        information_collected=[
            "force_restart_applied", "issue_confirmed_resolved",
        ],
        conversation_turn=2,
        kb_domain="black_screen",
        true_intent="black_screen",
        true_issue="issue_resolved",
        required_information=[],
        ideal_action="COMPLETE",
        ideal_clarification="",
        acceptable_actions=["COMPLETE"],
        acceptable_clarification_topics=[],
        resolution_path="close_ticket",
        should_not_do=[
            "ASK_CLARIFICATION", "SEARCH_KB", "PROVIDE_STEP",
        ],
    )

    # ================================================================
    # CATEGORY 7: EDGE CASES (4 scenarios)
    # Unknown device, very short message, unrelated request, nonsense
    # ================================================================

    add(
        category="edge_case_unknown_device",
        customer_message=(
            "My Xiaomi Redmi Note 13 screen is cracked and won't "
            "respond to touch. I have insurance. Can I file a claim?"
        ),
        device_type="Xiaomi Redmi Note 13",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="damaged_screen",
        true_intent="damaged_screen",
        true_issue="cracked_screen_unknown_device",
        required_information=["device_coverage_verification"],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Verify if Xiaomi device is covered under their insurance plan"
        ),
        acceptable_actions=["ASK_CLARIFICATION", "ROUTE_CLAIM"],
        acceptable_clarification_topics=[
            "insurance_coverage", "device_eligibility",
        ],
        resolution_path="coverage_verification",
        should_not_do=["COMPLETE", "PROVIDE_STEP"],
    )

    add(
        category="edge_case_short_message",
        customer_message="Phone broken",
        device_type=None,
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="claim_routing",
        true_intent="claim_routing",
        true_issue="unknown_issue",
        required_information=[
            "device_model", "damage_type", "symptoms",
            "insurance_status",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Ask what device, what happened, and what broken means"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "device_model", "damage_description", "what_happened",
        ],
        resolution_path="diagnostics",
        should_not_do=["COMPLETE", "PROVIDE_STEP", "ROUTE_CLAIM"],
    )

    add(
        category="edge_case_unrelated",
        customer_message=(
            "Can you help me set up my email on my new iPhone 15? I "
            "just got it and I want to add my Gmail and work Outlook "
            "accounts."
        ),
        device_type="iPhone 15",
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="data_transfer",
        true_intent="data_transfer",
        true_issue="email_setup_not_support_scope",
        required_information=[],
        ideal_action="PROVIDE_STEP",
        ideal_clarification="",
        acceptable_actions=["PROVIDE_STEP", "SEARCH_KB", "COMPLETE"],
        acceptable_clarification_topics=[],
        resolution_path="email_setup_guide",
        should_not_do=["ROUTE_CLAIM", "ESCALATE"],
    )

    add(
        category="edge_case_garbled",
        customer_message=(
            "phone no work screen thing charge also not wifi help plz "
            "samsung"
        ),
        device_type=None,
        issue_category=None,
        information_collected=[],
        conversation_turn=0,
        kb_domain="claim_routing",
        true_intent="claim_routing",
        true_issue="unclear_multiple_symptoms",
        required_information=[
            "device_model", "primary_issue", "symptom_details",
        ],
        ideal_action="ASK_CLARIFICATION",
        ideal_clarification=(
            "Ask customer to describe the main problem one at a time"
        ),
        acceptable_actions=["ASK_CLARIFICATION"],
        acceptable_clarification_topics=[
            "primary_issue", "device_model", "symptoms",
        ],
        resolution_path="diagnostics",
        should_not_do=["COMPLETE", "PROVIDE_STEP", "ROUTE_CLAIM"],
    )

    return scenarios


def main():
    scenarios = build_challenge_scenarios()
    print(f"Generated {len(scenarios)} challenge scenarios")

    # Print category breakdown
    from collections import Counter
    categories = Counter(s["challenge_category"] for s in scenarios)
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Print action distribution
    actions = Counter(
        s["hidden_ground_truth"]["ideal_action"] for s in scenarios
    )
    print("\nAction distribution:")
    for action in [
        "ASK_CLARIFICATION", "SEARCH_KB", "PROVIDE_STEP",
        "ROUTE_CLAIM", "ESCALATE", "COMPLETE",
    ]:
        count = actions.get(action, 0)
        print(f"  {action}: {count}")

    # Save
    base = os.path.join(os.path.dirname(__file__), "..", "data", "challenge")
    os.makedirs(base, exist_ok=True)

    out_path = os.path.join(base, "scenarios.json")
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
