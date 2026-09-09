#!/usr/bin/env python3
"""Generate BALANCED synthetic customer support scenarios for AURA-Lite.

Creates ~450 scenarios across 10 device issue domains with controlled
action distribution for RL post-training. Each action type is represented
at target frequencies to prevent reward model collapse.

Target action distribution:
    ASK_CLARIFICATION: ~25-30%
    SEARCH_KB:         ~20%
    PROVIDE_STEP:      ~20%
    ROUTE_CLAIM:       ~15%
    ESCALATE:          ~5-8%
    COMPLETE:          ~5-8%

Usage:
    python scripts/generate_scenarios.py
"""

import json
import os
import random
import hashlib
from typing import List, Dict, Tuple
from collections import Counter

SEED = 42
random.seed(SEED)

# --- Domain definitions ---

DEVICES = [
    "iPhone 14", "iPhone 15", "iPhone 15 Pro", "iPhone 16",
    "Samsung Galaxy S23", "Samsung Galaxy S24", "Samsung Galaxy A54",
    "Google Pixel 8", "Google Pixel 7", "iPad Air",
    "iPad Pro", "Samsung Galaxy Tab S9", "Motorola Edge",
]

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

# --- Scenario templates per domain ---
# Each template: messages, ambiguity, difficulty, template_family, gt
#
# IMPORTANT: templates are spread across all 6 action types.
# Each domain has templates for ASK + at least 1-2 other actions.
# Cross-domain templates (escalate, complete) appear in multiple domains.

TEMPLATES = {
    # ===== BLACK SCREEN =====
    "black_screen": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My phone isn't working. Screen is dark I think.",
                "Something's wrong with my {device}. Can't really see the screen properly.",
                "My {device} is acting up. The display looks off, maybe black or very dim.",
                "Phone screen issue. It went dark or something. Not sure what happened.",
                "I think my {device} screen might be broken? It's really dark.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "black_screen_ambiguous_ask",
            "gt": {
                "true_issue": "software_crash",
                "required_information": [
                    "physical_damage_check", "recent_events",
                    "screen_completely_black_or_dim", "device_model_confirm",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "physical_damage", "screen_state", "recent_events",
                ],
                "resolution_path": "force_restart",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM", "ESCALATE"],
            },
        },
        # ASK_CLARIFICATION - misleading
        {
            "messages": [
                "I dropped my {device} and now the screen is black. I think the battery died.",
                "Screen went black after I got it wet. Probably just needs charging right?",
                "My kid threw my {device} and the screen is black. Is there a way to restart it?",
            ],
            "ambiguity": "misleading",
            "difficulty": "hard",
            "template_family": "black_screen_misleading_ask",
            "gt": {
                "true_issue": "physical_damage_possible",
                "required_information": [
                    "physical_damage_check", "drop_details",
                    "visible_cracks", "water_exposure_details",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "physical_damage", "visible_cracks", "drop_height",
                ],
                "resolution_path": "damage_assessment",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # SEARCH_KB - clear enough to search
        {
            "messages": [
                "My {device} screen went completely black. It was working fine this morning and suddenly just went dark. No drops or water damage.",
                "The screen on my {device} is totally black. Phone vibrates so it's on. Never dropped it. Just stopped showing anything after the last update.",
                "My {device} display is completely black since the iOS/Android update last night. No physical damage at all.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "black_screen_clear_search",
            "gt": {
                "true_issue": "software_crash",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "force_restart",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM"],
            },
        },
        # PROVIDE_STEP - enough info to give first step
        {
            "messages": [
                "My {device} screen is black but it vibrates when I get calls. No damage, no water. I already tried pressing the power button. What do I do next?",
                "Screen went black on my {device}. No cracks, no drops. Power button doesn't help. I need the force restart steps.",
                "My {device} froze and went to a black screen. It's not physically damaged. I've tried holding the power button for 10 seconds but nothing happened.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "black_screen_clear_step",
            "gt": {
                "true_issue": "software_crash",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "force_restart",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM", "ESCALATE"],
            },
        },
    ],

    # ===== CHARGING =====
    "charging": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My phone charges really slow. Takes forever.",
                "Something is wrong with my {device} battery I think. It's not charging well.",
                "Charging issues with my {device}. It's weird.",
                "Phone battery situation. Charging is off somehow.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "charging_ambiguous_ask",
            "gt": {
                "true_issue": "slow_charging",
                "required_information": [
                    "cable_type", "charger_wattage", "charging_speed_details",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "cable_type", "charger_type", "charging_behavior",
                ],
                "resolution_path": "charger_upgrade_or_port_clean",
                "should_not_do": ["ROUTE_CLAIM", "COMPLETE"],
            },
        },
        # ASK_CLARIFICATION - misleading
        {
            "messages": [
                "My {device} got wet and now it won't charge. The screen also flickers.",
                "Dropped my {device} in water and charging stopped. I put it in rice.",
            ],
            "ambiguity": "misleading",
            "difficulty": "hard",
            "template_family": "charging_misleading_ask",
            "gt": {
                "true_issue": "liquid_damage_charging",
                "required_information": [
                    "water_exposure_duration", "drying_method", "visible_damage",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION", "ROUTE_CLAIM"],
                "acceptable_clarification_topics": [
                    "water_exposure", "current_state", "insurance_coverage",
                ],
                "resolution_path": "liquid_damage_claim",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
        # SEARCH_KB - tried multiple cables, enough info
        {
            "messages": [
                "My {device} won't charge at all. I've tried two different cables and three outlets. Port looks clean.",
                "Plugged in my {device} but battery isn't going up. Tested the cable on another phone and it works. Port looks fine.",
                "{device} won't take a charge. Tried OEM cable and a third-party one. Both work on my other phone.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "charging_clear_search",
            "gt": {
                "true_issue": "charging_port_or_software",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": ["port_debris", "cable_type"],
                "resolution_path": "port_cleaning_or_restart",
                "should_not_do": ["COMPLETE", "ESCALATE"],
            },
        },
        # PROVIDE_STEP - clear problem, give troubleshooting step
        {
            "messages": [
                "My {device} stopped charging. Port looks like it has lint in it. What should I do?",
                "I can see debris in the charging port of my {device}. Phone won't charge. How do I clean it safely?",
                "Charging port on my {device} seems clogged. Tried blowing into it. Still won't charge.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "charging_clear_step",
            "gt": {
                "true_issue": "charging_port_debris",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "port_cleaning",
                "should_not_do": ["ROUTE_CLAIM", "COMPLETE", "ESCALATE"],
            },
        },
    ],

    # ===== BATTERY DRAIN =====
    "battery_drain": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "Phone dies fast. Might be the battery?",
                "My {device} keeps turning off. Battery related maybe.",
                "Something wrong with my phone. Runs out of juice too quick.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "battery_ambiguous_ask",
            "gt": {
                "true_issue": "excessive_drain",
                "required_information": [
                    "battery_health_percent", "usage_pattern",
                    "shutdown_behavior", "device_age",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "battery_health", "usage_pattern", "shutdown_details",
                ],
                "resolution_path": "battery_diagnostics",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM"],
            },
        },
        # SEARCH_KB - clear drain with details
        {
            "messages": [
                "My {device} battery drains super fast. Goes from 100% to 20% in 3 hours. Battery health is at 87%. No new apps installed.",
                "{device} battery life is terrible lately. Has to be charged 3 times a day. Battery health shows 85%. Started after the last software update.",
                "Battery on my {device} barely lasts half a day now. Battery health is 90% so that's fine. It started draining fast about a week ago.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "battery_clear_search",
            "gt": {
                "true_issue": "excessive_drain",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": ["app_usage", "recent_updates"],
                "resolution_path": "background_app_audit",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
            },
        },
        # PROVIDE_STEP - clear enough to give a step
        {
            "messages": [
                "My {device} battery drains in 4 hours. I checked and battery health is 92%. I have lots of apps running. What should I do first?",
                "{device} battery dies by noon. Health is good at 89%. I think background apps are killing it. How do I check?",
                "Battery on my {device} drains fast. Battery health: 94%. I know something is running in the background. Walk me through fixing it.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "battery_clear_step",
            "gt": {
                "true_issue": "background_app_drain",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "background_app_audit",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE", "COMPLETE"],
            },
        },
        # ROUTE_CLAIM - battery health critically low
        {
            "messages": [
                "My {device} battery health is at 62%. It dies in under 2 hours. I have insurance. Need a replacement battery or phone.",
                "{device} battery is shot. Health says 58%. Can barely make it through a phone call. I'd like to file a claim.",
            ],
            "ambiguity": "clear",
            "difficulty": "medium",
            "template_family": "battery_clear_claim",
            "gt": {
                "true_issue": "battery_end_of_life",
                "required_information": ["insurance_verification"],
                "ideal_action": "ROUTE_CLAIM",
                "acceptable_actions": ["ROUTE_CLAIM", "ASK_CLARIFICATION"],
                "acceptable_clarification_topics": ["insurance_status", "purchase_date"],
                "resolution_path": "battery_replacement_claim",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
    ],

    # ===== ACTIVATION =====
    "activation": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My new phone doesn't work. Just got it.",
                "Can't use my {device}. Something about activation or setup.",
                "My {device} shows some error about activation. Not sure what to do.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "activation_ambiguous_ask",
            "gt": {
                "true_issue": "activation_unclear",
                "required_information": [
                    "carrier_name", "sim_type", "error_message",
                    "new_or_replacement", "activation_lock_status",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "error_message", "carrier", "new_or_replacement",
                ],
                "resolution_path": "activation_diagnostics",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP", "ESCALATE"],
            },
        },
        # SEARCH_KB - clear activation issue
        {
            "messages": [
                "Just got a new {device} and can't activate it. I'm on T-Mobile with a physical SIM. It says 'SIM not supported'.",
                "Trying to activate my {device} on Verizon. eSIM transfer from old phone. Getting 'activation error' message.",
                "New {device} won't connect to AT&T. Physical SIM from my old phone. Error says 'no service'.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "activation_clear_search",
            "gt": {
                "true_issue": "sim_activation_failure",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "carrier_activation",
                "should_not_do": ["ROUTE_CLAIM", "COMPLETE"],
            },
        },
        # PROVIDE_STEP - eSIM transfer walkthrough
        {
            "messages": [
                "I need to transfer my eSIM from my old {device} to the new one. Both phones are here. Where do I start?",
                "Switching to a new {device}. Need to move my eSIM over. I'm on Verizon. Walk me through it.",
                "Got my new {device}. Old phone has eSIM on T-Mobile. Need step-by-step to transfer it.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "activation_clear_step",
            "gt": {
                "true_issue": "esim_transfer",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "esim_transfer_guide",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE", "COMPLETE"],
            },
        },
        # ESCALATE - activation lock, possible stolen device
        {
            "messages": [
                "I bought a used {device} and it has an activation lock. The seller won't respond. I can't use the phone at all.",
                "My {device} has an iCloud activation lock. I'm the owner but I forgot my password and recovery email is old. Apple support couldn't help.",
            ],
            "ambiguity": "clear",
            "difficulty": "hard",
            "template_family": "activation_escalate",
            "gt": {
                "true_issue": "activation_lock_unresolvable",
                "required_information": ["proof_of_ownership"],
                "ideal_action": "ESCALATE",
                "acceptable_actions": ["ESCALATE", "ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "proof_of_purchase", "previous_owner_contact",
                ],
                "resolution_path": "escalate_to_specialist",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE", "ROUTE_CLAIM"],
            },
        },
    ],

    # ===== WIFI / NETWORK =====
    "wifi_network": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "Internet isn't working on my phone. Not sure if it's Wi-Fi or data.",
                "My {device} has connectivity issues. Can't load anything.",
                "My {device} won't go online. Something is wrong with the connection.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "wifi_ambiguous_ask",
            "gt": {
                "true_issue": "connectivity_unclear",
                "required_information": [
                    "wifi_or_cellular", "other_devices_status",
                    "error_messages", "recent_changes",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "wifi_or_cellular", "other_devices", "error_details",
                ],
                "resolution_path": "connectivity_diagnostics",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM"],
            },
        },
        # PROVIDE_STEP - clear wifi issue
        {
            "messages": [
                "My {device} won't connect to Wi-Fi. It sees the network but fails to join. Other devices connect fine.",
                "Wi-Fi keeps disconnecting on my {device}. Works fine on other devices at home. Password is correct.",
                "{device} shows Wi-Fi connected but no internet. Every other device in my house works. Already tried airplane mode.",
                "Can't get Wi-Fi on my {device}. Password is correct, other phones connect. It just won't join.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "wifi_clear_step",
            "gt": {
                "true_issue": "wifi_connection_failure",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": ["error_message", "router_status"],
                "resolution_path": "network_reset",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
            },
        },
        # SEARCH_KB - specific wifi error
        {
            "messages": [
                "My {device} says 'Unable to join network' for my home Wi-Fi. Started after the latest update. Other networks work fine.",
                "Wi-Fi on my {device} drops every few minutes. Only happens on my home network. Getting 'authentication error'.",
                "{device} keeps getting 'IP address conflict' when connecting to Wi-Fi. Never had this before.",
            ],
            "ambiguity": "clear",
            "difficulty": "medium",
            "template_family": "wifi_clear_search",
            "gt": {
                "true_issue": "wifi_specific_error",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "forget_network_reconnect",
                "should_not_do": ["ROUTE_CLAIM", "COMPLETE"],
            },
        },
    ],

    # ===== DATA TRANSFER =====
    "data_transfer": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My stuff didn't transfer over. Photos are missing from my new phone.",
                "Transfer didn't work completely. Lost some contacts.",
                "Data transfer was incomplete. Not sure what's missing on my new {device}.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "data_transfer_ambiguous_ask",
            "gt": {
                "true_issue": "incomplete_transfer",
                "required_information": [
                    "transfer_method_used", "what_is_missing",
                    "old_device_available", "backup_status",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "transfer_method", "missing_data", "backup_availability",
                ],
                "resolution_path": "retry_transfer",
                "should_not_do": ["COMPLETE", "ROUTE_CLAIM"],
            },
        },
        # SEARCH_KB - clear transfer need
        {
            "messages": [
                "I need to transfer data from my old {device} to a new iPhone. Both phones are here. Old one is Android.",
                "Switching from Samsung to {device}. Need to move photos, contacts, and messages. Both phones are charged and ready.",
                "Got a new {device}. Old phone is a Pixel 7. Want to transfer everything. What's the best method?",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "data_transfer_clear_search",
            "gt": {
                "true_issue": "data_migration",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "guided_transfer",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
            },
        },
        # PROVIDE_STEP - iCloud backup restore
        {
            "messages": [
                "I have an iCloud backup from my old iPhone. Just got my new {device}. How do I restore it?",
                "New {device} is asking me about restoring from iCloud backup during setup. I have a backup from yesterday. Walk me through it.",
                "Want to restore my new {device} from iCloud. I already backed up my old phone. What do I do on the setup screen?",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "data_transfer_clear_step",
            "gt": {
                "true_issue": "icloud_restore",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "icloud_backup_restore",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE", "COMPLETE"],
            },
        },
    ],

    # ===== DAMAGED SCREEN =====
    "damaged_screen": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My screen looks weird. Not sure if it's cracked or just a display issue.",
                "There's something on my {device} screen. Lines or cracks, can't tell.",
                "Phone screen has issues. Might be damaged or a software thing.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "damaged_screen_ambiguous_ask",
            "gt": {
                "true_issue": "screen_damage_unclear",
                "required_information": [
                    "visible_cracks", "touch_response", "drop_history",
                    "screen_appearance_details",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "visible_damage", "touch_working", "drop_history",
                ],
                "resolution_path": "damage_assessment",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # ROUTE_CLAIM - clear damage
        {
            "messages": [
                "I cracked the screen on my {device}. It still works but there's a big crack across it. I have insurance.",
                "My {device} screen is shattered. Dropped it on concrete. Touch still works partially. Need to file a claim.",
                "Screen on my {device} has cracks all over it. Need to get it fixed or replaced. I'm insured.",
                "Dropped my {device} and the screen cracked badly. Glass is coming off. Want to start a claim.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "damaged_screen_clear_claim",
            "gt": {
                "true_issue": "cracked_screen",
                "required_information": [],
                "ideal_action": "ROUTE_CLAIM",
                "acceptable_actions": ["ROUTE_CLAIM"],
                "acceptable_clarification_topics": [],
                "resolution_path": "screen_replacement_claim",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # SEARCH_KB - screen issues that might not be damage
        {
            "messages": [
                "My {device} screen has green lines running across it. No drops or damage. It appeared out of nowhere.",
                "Display on my {device} is flickering. Never dropped it. Screen looks physically fine but keeps flashing.",
                "My {device} has dead pixels in the corner. No physical damage. Is this a defect?",
            ],
            "ambiguity": "clear",
            "difficulty": "medium",
            "template_family": "damaged_screen_clear_search",
            "gt": {
                "true_issue": "display_defect",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "display_calibration",
                "should_not_do": ["ROUTE_CLAIM", "COMPLETE"],
            },
        },
        # ESCALATE - severe damage, safety risk
        {
            "messages": [
                "My {device} screen is shattered and I can see the battery underneath. It's also getting really hot. I'm worried it might be dangerous.",
                "The screen on my {device} is so broken that sharp glass is exposed. I cut my finger on it. Is this a safety hazard?",
            ],
            "ambiguity": "clear",
            "difficulty": "hard",
            "template_family": "damaged_screen_escalate",
            "gt": {
                "true_issue": "severe_damage_safety",
                "required_information": [],
                "ideal_action": "ESCALATE",
                "acceptable_actions": ["ESCALATE", "ROUTE_CLAIM"],
                "acceptable_clarification_topics": [],
                "resolution_path": "safety_escalation",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
    ],

    # ===== LOST / STOLEN =====
    "lost_stolen": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "I can't find my phone. I might have left it somewhere or it could be stolen.",
                "My {device} is missing. Not sure what happened to it.",
                "Haven't seen my {device} since yesterday. Not sure if I lost it or someone took it.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "lost_stolen_ambiguous_ask",
            "gt": {
                "true_issue": "missing_device_unclear",
                "required_information": [
                    "last_known_location", "lost_or_stolen",
                    "find_my_attempted", "time_since_missing",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "last_location", "lost_or_stolen", "find_my_status",
                ],
                "resolution_path": "locate_then_decide",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # ROUTE_CLAIM - clearly stolen
        {
            "messages": [
                "My {device} was stolen from my car. Window was smashed. I already filed a police report. Need to file an insurance claim.",
                "Someone pickpocketed my {device} on the subway. I have the police report number. I need a replacement through my insurance.",
                "My {device} was stolen. I've already locked it with Find My and filed a police report. Ready to start the claim process.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "lost_stolen_clear_claim",
            "gt": {
                "true_issue": "stolen_device",
                "required_information": [],
                "ideal_action": "ROUTE_CLAIM",
                "acceptable_actions": ["ROUTE_CLAIM"],
                "acceptable_clarification_topics": [],
                "resolution_path": "theft_claim",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
        # PROVIDE_STEP - lost, help locate
        {
            "messages": [
                "I lost my {device} somewhere in my house. It's on silent. How do I find it?",
                "Can't find my {device}. I know it's nearby because it was here an hour ago. How do I make it ring?",
                "I misplaced my {device}. It's definitely in my apartment. How do I use Find My to locate it?",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "lost_stolen_clear_step",
            "gt": {
                "true_issue": "misplaced_device",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "find_my_device",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
            },
        },
    ],

    # ===== LIQUID DAMAGE =====
    "liquid_damage": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "My {device} stopped working after it got a little wet. Maybe just splash.",
                "Phone might have gotten wet. It's not working right.",
                "My {device} was near water and now it's acting weird. Screen glitches.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "liquid_ambiguous_ask",
            "gt": {
                "true_issue": "possible_liquid_damage",
                "required_information": [
                    "water_exposure_details", "symptoms",
                    "time_since_exposure", "insurance_status",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "exposure_details", "symptoms", "insurance",
                ],
                "resolution_path": "damage_assessment",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # ROUTE_CLAIM - clear water damage
        {
            "messages": [
                "I dropped my {device} in the pool. It was underwater for about a minute. Screen is dead now. I have insurance.",
                "My {device} went through the washing machine. It won't turn on at all. Need to file a claim.",
                "Spilled a full glass of water on my {device}. It turned off and won't come back on. Insurance claim needed.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "liquid_clear_claim",
            "gt": {
                "true_issue": "liquid_damage_total",
                "required_information": [],
                "ideal_action": "ROUTE_CLAIM",
                "acceptable_actions": ["ROUTE_CLAIM"],
                "acceptable_clarification_topics": [],
                "resolution_path": "liquid_damage_claim",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
        # PROVIDE_STEP - minor splash, immediate first aid
        {
            "messages": [
                "I just splashed some water on my {device} a few minutes ago. It still works but I want to make sure it's okay. What should I do right now?",
                "My {device} got rained on a little. It's still working fine. What steps should I take to prevent damage?",
                "A few drops of water got on my {device}. It seems okay but I'm worried. What should I do?",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "liquid_clear_step",
            "gt": {
                "true_issue": "minor_water_exposure",
                "required_information": [],
                "ideal_action": "PROVIDE_STEP",
                "acceptable_actions": ["PROVIDE_STEP", "SEARCH_KB"],
                "acceptable_clarification_topics": [],
                "resolution_path": "liquid_damage_first_aid",
                "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
            },
        },
        # SEARCH_KB - water damage uncertain
        {
            "messages": [
                "I dropped my {device} in water briefly. It works now but the speaker sounds muffled. No insurance. What are my options?",
                "My {device} got soaked in the rain yesterday. It works but the camera has fog inside. What can I do?",
            ],
            "ambiguity": "clear",
            "difficulty": "medium",
            "template_family": "liquid_clear_search",
            "gt": {
                "true_issue": "partial_liquid_damage",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "drying_procedure",
                "should_not_do": ["COMPLETE", "ESCALATE"],
            },
        },
    ],

    # ===== CLAIM ROUTING =====
    "claim_routing": [
        # ASK_CLARIFICATION - ambiguous
        {
            "messages": [
                "I think I need a new phone. Mine's messed up. Do I have insurance?",
                "My {device} is broken. What are my options?",
                "Can I get my phone replaced? Something is wrong with it.",
            ],
            "ambiguity": "ambiguous",
            "difficulty": "medium",
            "template_family": "claim_routing_ambiguous_ask",
            "gt": {
                "true_issue": "claim_eligibility_check",
                "required_information": [
                    "damage_type", "insurance_status",
                    "device_condition_details", "purchase_date",
                ],
                "ideal_action": "ASK_CLARIFICATION",
                "acceptable_actions": ["ASK_CLARIFICATION"],
                "acceptable_clarification_topics": [
                    "damage_description", "insurance_status", "what_happened",
                ],
                "resolution_path": "eligibility_assessment",
                "should_not_do": ["COMPLETE", "PROVIDE_STEP"],
            },
        },
        # ROUTE_CLAIM - clear claim request
        {
            "messages": [
                "I need to file a claim for my {device}. The screen is completely shattered. I have insurance through Asurion.",
                "How do I start a claim? My {device} is water damaged beyond repair. Insurance plan is active.",
                "I want to file an insurance claim. My {device} was stolen yesterday. I have the police report.",
                "Need a replacement {device}. Mine is broken beyond repair and I have coverage.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "claim_routing_clear_claim",
            "gt": {
                "true_issue": "claim_needed",
                "required_information": [],
                "ideal_action": "ROUTE_CLAIM",
                "acceptable_actions": ["ROUTE_CLAIM"],
                "acceptable_clarification_topics": [],
                "resolution_path": "claims_processing",
                "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
            },
        },
        # SEARCH_KB - wants to understand claim process
        {
            "messages": [
                "What's the deductible for replacing my {device}? I have an insurance plan. Screen is cracked.",
                "How long does the claim process take for a {device}? Wondering if I should bother.",
                "What's covered under my insurance for the {device}? I want to know before I file.",
            ],
            "ambiguity": "clear",
            "difficulty": "easy",
            "template_family": "claim_routing_clear_search",
            "gt": {
                "true_issue": "claim_info_request",
                "required_information": [],
                "ideal_action": "SEARCH_KB",
                "acceptable_actions": ["SEARCH_KB", "PROVIDE_STEP"],
                "acceptable_clarification_topics": [],
                "resolution_path": "claim_info_lookup",
                "should_not_do": ["ESCALATE", "COMPLETE"],
            },
        },
    ],
}

# ===== CROSS-DOMAIN TEMPLATES (ESCALATE + COMPLETE) =====
# These are appended to specific domains but represent distinct scenario types.

ESCALATE_TEMPLATES = [
    # Safety concerns
    {
        "domain": "battery_drain",
        "messages": [
            "My {device} battery is swelling. The back of the phone is bulging out. I'm scared it might explode.",
            "The {device} is extremely hot to the touch and smells like burning. Battery area is swollen. What do I do?",
        ],
        "ambiguity": "clear",
        "difficulty": "hard",
        "template_family": "battery_safety_escalate",
        "gt": {
            "true_issue": "battery_swelling_safety",
            "required_information": [],
            "ideal_action": "ESCALATE",
            "acceptable_actions": ["ESCALATE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "safety_escalation_battery",
            "should_not_do": ["PROVIDE_STEP", "COMPLETE", "SEARCH_KB"],
        },
    },
    # Repeated failed troubleshooting
    {
        "domain": "charging",
        "messages": [
            "I've called about my {device} charging issue 3 times now. I've tried everything your agents suggested: new cable, port cleaning, factory reset. Nothing works. I need to talk to a supervisor.",
            "This is my 4th time contacting support about my {device} not charging. I've followed every troubleshooting step. Please escalate this.",
        ],
        "ambiguity": "clear",
        "difficulty": "hard",
        "template_family": "charging_repeated_escalate",
        "gt": {
            "true_issue": "repeated_contact_unresolved",
            "required_information": [],
            "ideal_action": "ESCALATE",
            "acceptable_actions": ["ESCALATE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "supervisor_escalation",
            "should_not_do": ["PROVIDE_STEP", "COMPLETE", "SEARCH_KB"],
        },
    },
    # Complex multi-system issue
    {
        "domain": "wifi_network",
        "messages": [
            "My {device} has no Wi-Fi, no Bluetooth, and no cellular. All radios seem dead. I already did a factory reset and it didn't help.",
            "After the latest update, my {device} lost all wireless connectivity. Wi-Fi, cellular, Bluetooth - all gone. Factory reset didn't fix it.",
        ],
        "ambiguity": "clear",
        "difficulty": "hard",
        "template_family": "wifi_multi_system_escalate",
        "gt": {
            "true_issue": "all_radios_dead",
            "required_information": [],
            "ideal_action": "ESCALATE",
            "acceptable_actions": ["ESCALATE", "ROUTE_CLAIM"],
            "acceptable_clarification_topics": [],
            "resolution_path": "hardware_escalation",
            "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
        },
    },
    # Abusive prior agent
    {
        "domain": "claim_routing",
        "messages": [
            "The last agent hung up on me when I asked about my {device} claim. I've been trying to get this resolved for a week. I need a manager.",
            "Your previous rep was extremely rude about my {device} replacement. I want to speak to someone higher up.",
        ],
        "ambiguity": "clear",
        "difficulty": "hard",
        "template_family": "claim_bad_experience_escalate",
        "gt": {
            "true_issue": "customer_experience_escalation",
            "required_information": [],
            "ideal_action": "ESCALATE",
            "acceptable_actions": ["ESCALATE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "supervisor_escalation",
            "should_not_do": ["PROVIDE_STEP", "COMPLETE", "SEARCH_KB"],
        },
    },
    # Frustrated, nothing works
    {
        "domain": "black_screen",
        "messages": [
            "I've been dealing with this black screen on my {device} for 2 weeks. I've done force restart, DFU mode, factory reset, taken it to a repair shop. Nothing works. This is unacceptable.",
            "My {device} black screen issue is still not fixed after 3 support calls. Tried every solution. I need this escalated to engineering or management.",
        ],
        "ambiguity": "clear",
        "difficulty": "hard",
        "template_family": "black_screen_exhausted_escalate",
        "gt": {
            "true_issue": "chronic_unresolved_issue",
            "required_information": [],
            "ideal_action": "ESCALATE",
            "acceptable_actions": ["ESCALATE", "ROUTE_CLAIM"],
            "acceptable_clarification_topics": [],
            "resolution_path": "senior_tech_escalation",
            "should_not_do": ["PROVIDE_STEP", "COMPLETE"],
        },
    },
]

COMPLETE_TEMPLATES = [
    # Issue already resolved
    {
        "domain": "black_screen",
        "messages": [
            "Hey, I called earlier about my {device} black screen. I just wanted to let you know the force restart worked! Everything is back to normal. Thanks!",
            "Following up on my {device} issue. The screen is working again after the restart you suggested. Just wanted to confirm it's fixed.",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "black_screen_resolved_complete",
        "gt": {
            "true_issue": "issue_resolved",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "close_ticket",
            "should_not_do": ["ASK_CLARIFICATION", "SEARCH_KB", "PROVIDE_STEP"],
        },
    },
    # Customer confirms fix
    {
        "domain": "charging",
        "messages": [
            "Just wanted to say thanks. Cleaning the charging port on my {device} fixed the problem. It's charging normally now.",
            "The charging issue on my {device} is resolved! It was lint in the port like you said. All good now.",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "charging_resolved_complete",
        "gt": {
            "true_issue": "issue_resolved",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "close_ticket",
            "should_not_do": ["ASK_CLARIFICATION", "SEARCH_KB", "ROUTE_CLAIM"],
        },
    },
    # Simple info request fully answered
    {
        "domain": "data_transfer",
        "messages": [
            "I just wanted to check if my iCloud backup is complete before I trade in my {device}. I can see it says 'last backup: today'. Am I good to go?",
            "Quick question: I backed up my {device} to iCloud and it says backup successful. Is that all I need before switching phones?",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "data_transfer_info_complete",
        "gt": {
            "true_issue": "simple_confirmation",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE", "PROVIDE_STEP"],
            "acceptable_clarification_topics": [],
            "resolution_path": "confirm_and_close",
            "should_not_do": ["ROUTE_CLAIM", "ESCALATE"],
        },
    },
    # Wi-Fi fixed
    {
        "domain": "wifi_network",
        "messages": [
            "The network reset fixed my {device} Wi-Fi issue! It's connected and working great now. Thanks for the help.",
            "Following up: forgetting the network and reconnecting worked on my {device}. Wi-Fi is stable now.",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "wifi_resolved_complete",
        "gt": {
            "true_issue": "issue_resolved",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "close_ticket",
            "should_not_do": ["ASK_CLARIFICATION", "SEARCH_KB", "PROVIDE_STEP"],
        },
    },
    # Battery drain fixed
    {
        "domain": "battery_drain",
        "messages": [
            "Update on my {device} battery issue: I disabled background app refresh and it's lasting all day now. Problem solved!",
            "Wanted to let you know turning off location services for some apps fixed the battery drain on my {device}. Thanks!",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "battery_resolved_complete",
        "gt": {
            "true_issue": "issue_resolved",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "close_ticket",
            "should_not_do": ["ASK_CLARIFICATION", "SEARCH_KB", "ROUTE_CLAIM"],
        },
    },
    # Claim completed
    {
        "domain": "claim_routing",
        "messages": [
            "I received my replacement {device} today! Everything is set up and working. Thank you for processing the claim so quickly.",
            "Just confirming my claim for the {device} is done. Got the new phone, transferred my data. All good. Thanks!",
        ],
        "ambiguity": "clear",
        "difficulty": "easy",
        "template_family": "claim_completed_complete",
        "gt": {
            "true_issue": "claim_resolved",
            "required_information": [],
            "ideal_action": "COMPLETE",
            "acceptable_actions": ["COMPLETE"],
            "acceptable_clarification_topics": [],
            "resolution_path": "close_ticket",
            "should_not_do": ["ASK_CLARIFICATION", "ROUTE_CLAIM", "PROVIDE_STEP"],
        },
    },
]


def generate_scenario(
    scenario_id: str,
    domain: str,
    template: dict,
    message: str,
    device: str,
) -> dict:
    """Generate a single scenario from template."""
    msg = message.format(device=device)
    gt = template["gt"]

    return {
        "scenario_id": scenario_id,
        "ambiguity_level": template["ambiguity"],
        "difficulty": template["difficulty"],
        "template_family": template["template_family"],
        "state": {
            "customer_message": msg,
            "device_type": device if random.random() > 0.3 else None,
            "issue_category": None,
            "information_collected": [],
            "conversation_turn": 0,
            "available_kb_topics": random.sample(
                KB_TOPICS[domain],
                min(4, len(KB_TOPICS[domain])),
            ),
        },
        "hidden_ground_truth": {
            "true_intent": domain,
            "true_issue": gt["true_issue"],
            "required_information": gt["required_information"],
            "ideal_action": gt["ideal_action"],
            "ideal_clarification": (
                f"Clarification for {domain} issue"
            ),
            "acceptable_actions": gt["acceptable_actions"],
            "acceptable_clarification_topics": gt[
                "acceptable_clarification_topics"
            ],
            "resolution_path": gt["resolution_path"],
            "should_not_do": gt["should_not_do"],
        },
    }


def generate_all_scenarios() -> List[dict]:
    """Generate ~450 balanced scenarios across domains and action types.

    Strategy:
    - Phase 1: Domain templates x devices (action-balanced by template design)
    - Phase 2: Escalate cross-domain templates x devices
    - Phase 3: Complete cross-domain templates x devices
    - Phase 4: Boosting pass for underrepresented actions

    The template design ensures balanced action distribution by having
    each domain contribute to multiple action types.
    """
    scenarios = []
    counter = 0

    # Phase 1: Domain templates — each message x 2-3 device variants
    for domain, template_groups in TEMPLATES.items():
        for tg in template_groups:
            messages = tg["messages"]
            for msg in messages:
                n_devices = 2 if len(messages) >= 4 else 3
                sampled_devices = random.sample(
                    DEVICES, min(n_devices, len(DEVICES))
                )
                for device in sampled_devices:
                    counter += 1
                    sid = f"scn_{counter:04d}"
                    scenario = generate_scenario(
                        sid, domain, tg, msg, device
                    )
                    scenarios.append(scenario)

    # Phase 2: Escalate templates
    for tpl in ESCALATE_TEMPLATES:
        domain = tpl["domain"]
        for msg in tpl["messages"]:
            sampled_devices = random.sample(DEVICES, 2)
            for device in sampled_devices:
                counter += 1
                sid = f"scn_{counter:04d}"
                template_dict = {
                    "ambiguity": tpl["ambiguity"],
                    "difficulty": tpl["difficulty"],
                    "template_family": tpl["template_family"],
                    "gt": tpl["gt"],
                }
                scenario = generate_scenario(
                    sid, domain, template_dict, msg, device
                )
                scenarios.append(scenario)

    # Phase 3: Complete templates
    for tpl in COMPLETE_TEMPLATES:
        domain = tpl["domain"]
        for msg in tpl["messages"]:
            sampled_devices = random.sample(DEVICES, 3)
            for device in sampled_devices:
                counter += 1
                sid = f"scn_{counter:04d}"
                template_dict = {
                    "ambiguity": tpl["ambiguity"],
                    "difficulty": tpl["difficulty"],
                    "template_family": tpl["template_family"],
                    "gt": tpl["gt"],
                }
                scenario = generate_scenario(
                    sid, domain, template_dict, msg, device
                )
                # Complete scenarios should have turn > 0 (follow-up)
                scenario["state"]["conversation_turn"] = random.choice([1, 2])
                scenario["state"]["information_collected"] = random.sample(
                    [
                        "issue previously reported",
                        "troubleshooting steps provided",
                        "customer confirmed understanding",
                        "resolution applied",
                        "replacement shipped",
                        "claim processed",
                    ],
                    random.randint(1, 3),
                )
                scenarios.append(scenario)

    # Phase 4: Boost PROVIDE_STEP and SEARCH_KB with turn > 0 variants
    # (when info is already collected, model should ACT not ask)
    ask_scenarios = [
        s for s in scenarios
        if s["hidden_ground_truth"]["ideal_action"] == "ASK_CLARIFICATION"
    ]
    n_boost = min(30, len(ask_scenarios))
    for s in random.sample(ask_scenarios, n_boost):
        counter += 1
        variant = json.loads(json.dumps(s))
        variant["scenario_id"] = f"scn_{counter:04d}"
        variant["state"]["conversation_turn"] = 2
        gt = variant["hidden_ground_truth"]
        req = gt["required_information"]
        if len(req) >= 2:
            variant["state"]["information_collected"] = req[:-1]
        elif len(req) == 1:
            variant["state"]["information_collected"] = req[:]
        else:
            variant["state"]["information_collected"] = [
                "issue described clearly"
            ]
        # With info collected, shift ideal action
        if gt["true_intent"] in ("claim_routing", "damaged_screen",
                                  "lost_stolen", "liquid_damage"):
            gt["ideal_action"] = "ROUTE_CLAIM"
            gt["acceptable_actions"] = ["ROUTE_CLAIM", "SEARCH_KB"]
        else:
            gt["ideal_action"] = random.choice(
                ["SEARCH_KB", "PROVIDE_STEP"]
            )
            gt["acceptable_actions"] = ["SEARCH_KB", "PROVIDE_STEP"]
        variant["ambiguity_level"] = "clear"
        variant["difficulty"] = "medium"
        variant["template_family"] = (
            variant["template_family"] + "_boosted"
        )
        scenarios.append(variant)

    random.shuffle(scenarios)
    return scenarios


def split_scenarios(
    scenarios: List[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Stratified split by (action_type, domain), grouping same-template
    scenarios together to prevent leakage.

    Stratifies by action type first to ensure every action appears
    in every split, then by domain within each action type.
    """

    # Group scenarios by template_family
    template_groups: Dict[str, List[dict]] = {}
    for s in scenarios:
        key = s["template_family"]
        template_groups.setdefault(key, []).append(s)

    # Group template families by action type for stratified split
    by_action: Dict[str, List[List[dict]]] = {}
    for family, group in template_groups.items():
        action = group[0]["hidden_ground_truth"]["ideal_action"]
        by_action.setdefault(action, []).append(group)

    train, val, test = [], [], []
    for action, groups in by_action.items():
        random.shuffle(groups)
        n = len(groups)
        if n <= 2:
            # Very few groups: put all in train (can't split without
            # losing the action in a split)
            for g in groups:
                train.extend(g)
        elif n <= 4:
            # Small number: 1 val, 1 test, rest train
            for g in groups[:-2]:
                train.extend(g)
            val.extend(groups[-2])
            test.extend(groups[-1])
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            # Ensure at least 1 group for test
            if n_train + n_val >= n:
                n_train = n - 2
                n_val = 1

            for g in groups[:n_train]:
                train.extend(g)
            for g in groups[n_train:n_train + n_val]:
                val.extend(g)
            for g in groups[n_train + n_val:]:
                test.extend(g)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def check_leakage(
    train: List[dict],
    val: List[dict],
    test: List[dict],
) -> None:
    """Check for template-level near-duplicate leakage.
    Uses template_family field for grouping."""
    train_families = {s["template_family"] for s in train}
    val_families = {s["template_family"] for s in val}
    test_families = {s["template_family"] for s in test}

    val_leak = train_families & val_families
    test_leak = train_families & test_families

    if val_leak or test_leak:
        print(f"  WARNING: {len(val_leak)} val family leaks, "
              f"{len(test_leak)} test family leaks")
        if val_leak:
            print(f"    Val leaks: {val_leak}")
        if test_leak:
            print(f"    Test leaks: {test_leak}")
    else:
        print("  No template-level leakage detected.")


def print_action_distribution(
    scenarios: List[dict], label: str
) -> None:
    """Print action distribution for a split."""
    actions = Counter(
        s["hidden_ground_truth"]["ideal_action"]
        for s in scenarios
    )
    total = sum(actions.values())
    print(f"\n  {label} action distribution (n={total}):")
    for action in [
        "ASK_CLARIFICATION", "SEARCH_KB", "PROVIDE_STEP",
        "ROUTE_CLAIM", "ESCALATE", "COMPLETE",
    ]:
        count = actions.get(action, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {action:20s}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    scenarios = generate_all_scenarios()
    print(f"Generated {len(scenarios)} total scenarios")

    # Domain distribution
    domains = Counter(
        s["hidden_ground_truth"]["true_intent"]
        for s in scenarios
    )
    print("\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")

    # Overall action distribution
    print_action_distribution(scenarios, "OVERALL")

    # Ambiguity distribution
    ambiguity = Counter(s["ambiguity_level"] for s in scenarios)
    print(f"\nAmbiguity distribution:")
    for level, count in sorted(ambiguity.items()):
        print(f"  {level}: {count} ({count/len(scenarios)*100:.1f}%)")

    # Difficulty distribution
    difficulty = Counter(s["difficulty"] for s in scenarios)
    print(f"\nDifficulty distribution:")
    for level, count in sorted(difficulty.items()):
        print(f"  {level}: {count} ({count/len(scenarios)*100:.1f}%)")

    # Split
    train, val, test = split_scenarios(scenarios)
    print(f"\nSplits: train={len(train)}, val={len(val)}, "
          f"test={len(test)}")

    check_leakage(train, val, test)
    print_action_distribution(train, "TRAIN")
    print_action_distribution(val, "VAL")
    print_action_distribution(test, "TEST")

    # Save
    base = os.path.join(
        os.path.dirname(__file__), "..", "data"
    )
    os.makedirs(os.path.join(base, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base, "processed"), exist_ok=True)

    with open(os.path.join(base, "raw", "scenarios.json"), "w") as f:
        json.dump(scenarios, f, indent=2)

    with open(os.path.join(base, "processed", "train.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(base, "processed", "val.json"), "w") as f:
        json.dump(val, f, indent=2)

    with open(os.path.join(base, "processed", "test.json"), "w") as f:
        json.dump(test, f, indent=2)

    print(f"\nSaved to {base}/")


if __name__ == "__main__":
    main()
