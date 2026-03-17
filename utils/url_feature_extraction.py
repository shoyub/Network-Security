"""
URL Feature Extraction Utility for Phishing Detection
Extracts features from URLs to match the training dataset format
"""

import pandas as pd
import re
import urllib.parse
from urllib.parse import urlparse
import string


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""

    def __init__(self):
        """Initialize feature extractor with suspicious words and patterns"""
        self.suspicious_keywords = [
            'login',
            'sign',
            'account',
            'update',
            'confirm',
            'verify',
            'suspended',
            'urgent',
            'unusual',
            'activity',
            'click',
            'bank',
            'secure',
            'ebay',
            'amazon',
            'paypal',
            'apple',
            'google',
            'microsoft'
        ]

    def extract_features(self, url: str) -> dict:
        """
        Extract all features from a URL
        Returns a dictionary with feature names and values
        """
        features = {}
        
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Basic URL features
            features['URL_LENGTH'] = len(url)
            features['DOMAIN_LENGTH'] = len(parsed.netloc)
            features['PATH_LENGTH'] = len(parsed.path)
            
            # Count special characters
            features['SPECIAL_CHARS_COUNT'] = sum(
                1 for char in url if char in string.punctuation
            )
            
            # Protocol features
            protocol = parsed.scheme.lower()
            features['USES_HTTPS'] = 1 if protocol == 'https' else 0
            
            # Separator counts
            features['DOT_COUNT'] = url.count('.')
            features['DASH_COUNT'] = url.count('-')
            features['SLASH_COUNT'] = url.count('/')
            features['QUESTION_MARK_COUNT'] = url.count('?')
            features['AMPERSAND_COUNT'] = url.count('&')
            features['UNDERSCORE_COUNT'] = url.count('_')
            features['AT_SYMBOL_COUNT'] = url.count('@')
            features['COLON_COUNT'] = url.count(':')
            
            # Domain features
            domain = parsed.netloc
            features['SUBDOMAIN_COUNT'] = domain.count('.') - (1 if '.' in domain else 0)
            
            # Check for IP address in domain
            features['HAS_IP_ADDRESS'] = 1 if self._is_ip_address(domain) else 0
            
            # Check for suspicious keywords
            url_lower = url.lower()
            features['SUSPICIOUS_KEYWORD_COUNT'] = sum(
                1 for keyword in self.suspicious_keywords 
                if keyword in url_lower
            )
            
            # URL entropy (randomness measure)
            features['ENTROPY'] = self._calculate_entropy(url)
            
            # Digit count
            features['DIGIT_COUNT'] = sum(1 for char in url if char.isdigit())
            
            # Percentage of digits
            features['DIGIT_RATIO'] = (
                features['DIGIT_COUNT'] / features['URL_LENGTH'] 
                if features['URL_LENGTH'] > 0 else 0
            )
            
            # Percentage of special characters
            features['SPECIAL_CHAR_RATIO'] = (
                features['SPECIAL_CHARS_COUNT'] / features['URL_LENGTH']
                if features['URL_LENGTH'] > 0 else 0
            )
            
            # Check for query string
            features['HAS_QUERY_STRING'] = 1 if parsed.query else 0
            query_length = len(parsed.query) if parsed.query else 0
            features['QUERY_LENGTH'] = query_length
            
            # Check for fragments
            features['HAS_FRAGMENT'] = 1 if parsed.fragment else 0
            
            # Check for port number in URL
            features['HAS_PORT'] = 1 if parsed.port else 0
            
            # Count uppercase letters
            features['UPPERCASE_COUNT'] = sum(1 for char in url if char.isupper())
            features['UPPERCASE_RATIO'] = (
                features['UPPERCASE_COUNT'] / features['URL_LENGTH']
                if features['URL_LENGTH'] > 0 else 0
            )
            
            # Double slash count
            features['DOUBLE_SLASH_COUNT'] = url.count('//')
            
            # Check for www
            features['HAS_WWW'] = 1 if 'www' in domain else 0
            
            # Domain character count (letters only)
            domain_letters = sum(1 for char in domain if char.isalpha())
            features['DOMAIN_LETTER_COUNT'] = domain_letters
            
            # Check for suspicious TLDs
            tld = self._extract_tld(domain)
            features['TLD_LENGTH'] = len(tld)
            features['IS_COMMON_TLD'] = 1 if tld in [
                'com', 'org', 'net', 'edu', 'gov', 'mil', 'co', 'uk', 'de', 'fr'
            ] else 0
            
        except Exception as e:
            # Return zeros for invalid URLs
            print(f"Error extracting features: {e}")
            return self._default_features()
        
        return features

    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(pattern, domain))

    def _calculate_entropy(self, url: str) -> float:
        """Calculate Shannon entropy of URL"""
        import math
        
        if len(url) == 0:
            return 0
        
        char_count = {}
        for char in url:
            char_count[char] = char_count.get(char, 0) + 1
        
        entropy = 0
        for count in char_count.values():
            probability = count / len(url)
            entropy -= probability * math.log2(probability)
        
        return round(entropy, 4)

    def _extract_tld(self, domain: str) -> str:
        """Extract top-level domain from domain name"""
        parts = domain.split('.')
        if len(parts) > 0:
            return parts[-1].lower()
        return ''

    def _default_features(self) -> dict:
        """Return default zero features for error handling"""
        return {
            'URL_LENGTH': 0,
            'DOMAIN_LENGTH': 0,
            'PATH_LENGTH': 0,
            'SPECIAL_CHARS_COUNT': 0,
            'USES_HTTPS': 0,
            'DOT_COUNT': 0,
            'DASH_COUNT': 0,
            'SLASH_COUNT': 0,
            'QUESTION_MARK_COUNT': 0,
            'AMPERSAND_COUNT': 0,
            'UNDERSCORE_COUNT': 0,
            'AT_SYMBOL_COUNT': 0,
            'COLON_COUNT': 0,
            'SUBDOMAIN_COUNT': 0,
            'HAS_IP_ADDRESS': 0,
            'SUSPICIOUS_KEYWORD_COUNT': 0,
            'ENTROPY': 0,
            'DIGIT_COUNT': 0,
            'DIGIT_RATIO': 0,
            'SPECIAL_CHAR_RATIO': 0,
            'HAS_QUERY_STRING': 0,
            'QUERY_LENGTH': 0,
            'HAS_FRAGMENT': 0,
            'HAS_PORT': 0,
            'UPPERCASE_COUNT': 0,
            'UPPERCASE_RATIO': 0,
            'DOUBLE_SLASH_COUNT': 0,
            'HAS_WWW': 0,
            'DOMAIN_LETTER_COUNT': 0,
            'TLD_LENGTH': 0,
            'IS_COMMON_TLD': 0,
        }

    def extract_features_dataframe(self, url: str) -> pd.DataFrame:
        """
        Extract features from URL and return as pandas DataFrame
        This format is suitable for model prediction
        """
        features = self.extract_features(url)
        return pd.DataFrame([features])


# Example usage:
if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://suspicious-bank-login-verify.com/account/update?user=admin",
        "https://paypal.com",
    ]
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = extractor.extract_features(url)
        print(f"Features: {features}")
