def extract_fields(text: str) -> dict:
    low = text.lower()

    # --- VRM ---
    vrm_match = VRM.search(text)
    vrm = vrm_match.group(0) if vrm_match else None

    # --- Contravention date ---
    contravention_date = None
    pref = re.search(r"(?:contravention\s*date|date\s*of\s*contravention)[:\s-]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.IGNORECASE)
    if pref:
        contravention_date = pref.group(1)
    else:
        m = DATE_DMY.search(text)
        if m:
            contravention_date = m.group(1)

    # --- Contravention code/type ---
    contravention_code = None
    contravention_type = None
    cm = CODE.search(text)
    if cm:
        contravention_code = cm.group(0)
        contravention_type = CONTRAVENTION_TYPES.get(contravention_code, "Other")

    # Override with explicit reason if found
    contravention_reason = None
    for line in text.splitlines():
        rm = REASON.search(line)
        if rm:
            contravention_reason = re.split(r"[.\(\[]", rm.group(1).strip())[0].strip()
            break
    if contravention_reason:
        contravention_type = contravention_reason

    # --- PCN number ---
    pcn_number = None
    pm = PCN_NO.search(text)
    if pm:
        pcn_number = pm.group(1).strip("-")

    # --- Location ---
    location = None
    lm = LOCATION.search(text)
    if lm:
        location = (lm.group("loc1") or lm.group("loc2") or "").strip(" :-").strip()
        if len(location) > 120:
            location = None

    # --- Authority ---
    authority = None
    am = re.search(r"(?:Issued\s*by|Enforcement\s*Authority)[:\s-]*(.+)", text, re.IGNORECASE)
    if am:
        authority = re.split(r"[.\n\r]", am.group(1).strip())[0].strip()
    else:
        am = AUTHORITY.search(text)
        if am:
            authority = am.group(0).strip()

    # --- Fine amounts ---
    all_amounts = [float(m.group(1).replace(",", "")) for m in MONEY.finditer(text)]
    fine_amount_discounted = min(all_amounts) if len(all_amounts) >= 2 else None
    fine_amount_full = max(all_amounts) if all_amounts else None

    # --- Due date for payment ---
    due_date = None
    dm = re.search(
        r"(?:paid\s*by|before|no later than|must be paid by)[:\s-]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text,
        re.IGNORECASE,
    )
    if dm:
        due_date = dm.group(1)

    return {
        "vrm": vrm,
        "contravention_date": contravention_date,
        "contravention_code": contravention_code,
        "contravention_type": contravention_type,
        "pcn_number": pcn_number,
        "location": location,
        "authority": authority,
        "fine_amount_discounted": fine_amount_discounted,
        "fine_amount_full": fine_amount_full,
        "due_date": due_date,
    }
