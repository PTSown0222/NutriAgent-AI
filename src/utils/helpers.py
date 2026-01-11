import re
import unicodedata

# BẢNG MÃ FULL TCVN3 -> UNICODE (Đầy đủ cả Hoa/Thường và ký tự PDF lỗi)
TCVN3_TO_UNICODE = {
    # --- Nguyên âm thường ---
    'µ': 'à', '¸': 'á', '¶': 'ả', '·': 'ã', '¹': 'ạ',
    '¨': 'ă', '¾': 'ắ', '»': 'ằ', '¼': 'ẳ', '½': 'ẵ', 'Æ': 'ặ',
    '©': 'â', 'Ê': 'ấ', 'Ç': 'ầ', 'Ë': 'ẩ', 'É': 'ẫ', 'È': 'ậ',
    '®': 'đ',
    'Ð': 'é', 'Ì': 'ẻ', 'Î': 'ẽ', 'Ï': 'ẹ',
    'ª': 'ê', 'Õ': 'ế', 'Ò': 'ề', 'Ó': 'ể', 'Ô': 'ễ', 'Ö': 'ệ',
    '×': 'ì', 'Ý': 'í', 'Ø': 'ỉ', 'Ü': 'ĩ', 'Þ': 'ị',
    'ß': 'ò', 'ã': 'ó', 'á': 'ỏ', 'â': 'õ', 'ä': 'ọ',
    '«': 'ô', 'è': 'ộ', 'å': 'ồ', 'æ': 'ố', 'ç': 'ổ', 'é': 'ỗ',
    '¬': 'ư', 'ú': 'ử', 'ù': 'ứ', 'û': 'ữ', 'ü': 'ự', '÷': 'ư', # Có nhiều biến thể của ư
    'ê': 'ơ', 'í': 'ớ', 'ë': 'ờ', 'ì': 'ở', 'î': 'ỡ', 'ï': 'ợ',
    'ó': 'ù', 'ò': 'ú', 'ô': 'ủ', 'õ': 'ũ', 'ö': 'ụ',
    'ý': 'ỳ', 'þ': 'ỵ', 'ø': 'ừ', '÷': 'ự', 'û': 'ỷ', 'ü': 'ỹ',
    
    # --- Nguyên âm HOA (Thường bị thiếu trong bảng cũ) ---
    'µ': 'À', '¸': 'Á', '¶': 'Ả', '·': 'Ã', '¹': 'Ạ',
    '¡': 'Ă', '¾': 'Ắ', '»': 'Ằ', '¼': 'Ẳ', '½': 'Ẵ', 'Æ': 'Ặ',
    '¢': 'Â', 'Ê': 'Ấ', 'Ç': 'Ầ', 'Ë': 'Ẩ', 'É': 'Ẫ', 'È': 'Ậ',
    '§': 'Đ',
    '£': 'Ê', 'Õ': 'Ế', 'Ò': 'Ề', 'Ó': 'Ể', 'Ô': 'Ễ', 'Ö': 'Ệ',
    '¤': 'Ô', 'è': 'Ộ', 'å': 'Ồ', 'æ': 'Ố', 'ç': 'Ổ', 'é': 'Ỗ',
    '¥': 'Ơ', 'í': 'Ớ', 'ë': 'Ờ', 'ì': 'Ở', 'î': 'Ỡ', 'ï': 'Ợ',
    '¦': 'Ư', 'ú': 'Ử', 'ù': 'Ứ', 'û': 'Ữ', 'ü': 'Ự', 'ø': 'Ừ',
    
    # --- Ký tự lỗi PDF đặc thù (Do font map sai) ---
    '3/4': 'ư', 
    '1/4': 'ở',
    '1/2': 'ả',
    '–': '', # Gạch nối dài thường gây lỗi
}

def clean_broken_layout(text: str) -> str:
    """Hàn gắn các ký tự bị xuống dòng vô lý"""
    if not text: return ""
    # M \n a \n g \n i \n ê -> Magiê
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])\n(?=[a-zA-Zà-ỹÀ-Ỹ])', '', text)
    # Xóa dấu gạch ngang dài thừa thãi (-----)
    text = re.sub(r'-{3,}', ' ', text)
    return text

def fix_encoding(text: str) -> str:
    """
    Hàm sửa lỗi encoding TCVN3 toàn diện.
    Chiến thuật: Thay thế ký tự -> Chuẩn hóa Unicode NFC
    """
    if not text: return ""

    # 1. Sửa layout trước
    text = clean_broken_layout(text)

    # 2. Bảo vệ đơn vị đo lường (để không bị map nhầm)
    # TCVN3 dùng 'µ' là 'à', nhưng trong khoa học 'µ' là micro
    # Logic: Nếu µ đứng trước g, l, m... thì giữ nguyên. Nếu µ đứng trong từ (vµng) thì đổi.
    text = text.replace("µg", "##MICRO_GRAM##")
    text = text.replace("mg", "##MILLI_GRAM##")

    # 3. Thay thế ký tự TCVN3 bằng Unicode
    # Chúng ta iter qua bảng map để replace (An toàn hơn translate với chuỗi dài như 3/4)
    for tcvn_char, unicode_char in TCVN3_TO_UNICODE.items():
        if tcvn_char in text:
            text = text.replace(tcvn_char, unicode_char)

    # 4. Khôi phục đơn vị đo
    text = text.replace("##MICRO_GRAM##", "µg")
    text = text.replace("##MILLI_GRAM##", "mg")

    # 5.Chuẩn hóa Unicode (NFC)
    # Bước này giúp gộp các ký tự rời rạc (a + `) thành 1 ký tự (à)
    text = unicodedata.normalize('NFC', text)

    # 6. Fix các lỗi chính tả phổ biến còn sót lại do PDF OCR sai
    # Chỉ giữ lại những từ cực kỳ đặc thù không thể map bằng bảng mã
    text = text.replace("B,nh", "Bánh")
    text = text.replace("m,y", "mây")
    text = text.replace("l,t", "lứt")
    text = text.replace("tî", "tẻ") # tî thường là tẻ trong 1 số font

    return text.strip()