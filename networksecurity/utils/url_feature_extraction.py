"""
URL Feature Extraction Utility for Phishing Detection
Extracts features from URLs to match the training dataset format (schema.yaml)
"""

import pandas as pd
import re
from urllib.parse import urlparse
import string


class URLFeatureExtractor:
    """Extract features from URLs matching the trained model's expected features"""

    def __init__(self):
        """Initialize with phishing patterns"""
        self.phishing_keywords = [
            'login', 'signin', 'sign-in', 'account', 'verify', 'confirm', 
            'update', 'urgent', 'suspended', 'bank', 'paypal', 'amazon', 
            'apple', 'secure', 'password', 'auth', 'submit', 'check', 'click',
            'href', 'redirect', 'action', 'validate'
        ]
        
        self.legitimate_tlds = ['com', 'org', 'net', 'edu', 'gov', 'co.uk', 'de', 'fr', 'io', 'dev']

    def calculate_phishing_score(self, url: str) -> float:
        """
        Calculate a heuristic phishing score (0-100) based on URL patterns
        Used to enhance model predictions when external data is unavailable
        """
        score = 0
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]
        
        # 1. Protocol (HTTPS is good, HTTP is suspicious)
        if parsed.scheme == 'http':
            score += 15
        
        # 2. Suspicious keywords in domain or path
        url_lower = url.lower()
        keyword_count = sum(1 for kw in self.phishing_keywords if kw in url_lower)
        score += min(keyword_count * 10, 40)  # Max 40
        
        # 3. IP address instead of domain
        if self._is_ip_address(domain):
            score += 25
        
        # 4. @ symbol (used to obfuscate domain)
        if '@' in url:
            score += 20
        
        # 5. Multiple dots in domain (subdomain abuse)
        if domain.count('.') > 2:
            score += 15
        
        # 6. Unusual TLD
        tld = self._extract_tld(domain)
        if tld and tld not in self.legitimate_tlds:
            score += 10
        
        # 7. Dash in domain (common in phishing)
        if '-' in domain:
            score += 8
        
        # 8. Very long URL (common obfuscation)
        if len(url) > 100:
            score += 10
        
        # 9. Multiple slashes (often redirection)
        if url.count('//') > 1:
            score += 15
        
        # 10. Query parameters with suspicious values
        if parsed.query:
            if any(kw in parsed.query.lower() for kw in ['redirect', 'url', 'link']):
                score += 12
        
        return min(score, 100)  # Cap at 100

    def extract_features(self, url: str) -> dict:
        """
        Extract features from URL matching schema.yaml columns
        Returns a dictionary with feature names exactly as expected by the model
        """
        features = {}
        
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # === EXTRACTABLE FROM URL ===
            
            # URL Length
            features['URL_Length'] = len(url)
            
            # Check for IP Address in domain
            features['having_IP_Address'] = 1 if self._is_ip_address(parsed.netloc) else 0
            
            # @ Symbol
            features['having_At_Symbol'] = 1 if '@' in url else 0
            
            # Double slash in middle (redirection)
            features['double_slash_redirecting'] = 1 if url.count('//') > 1 else 0
            
            # Prefix-Suffix (dash in domain)
            domain = parsed.netloc.split(':')[0]  # Remove port if exists
            features['Prefix_Suffix'] = 1 if '-' in domain else 0
            
            # Subdomains
            subdomain_count = domain.count('.')
            features['having_Sub_Domain'] = 1 if subdomain_count > 1 else 0
            
            # HTTPS
            features['SSLfinal_State'] = 1 if parsed.scheme == 'https' else 0
            
            # Port
            features['port'] = 1 if parsed.port is not None else 0
            
            # HTTPS token in domain (should not be there if legitimate)
            features['HTTPS_token'] = 1 if 'https' in domain.lower() else 0
            
            # Shortening service detection
            shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'short.link', 'is.gd']
            features['Shortining_Service'] = 1 if any(s in url.lower() for s in shorteners) else 0
            
            # Abnormal URL patterns (suspicious keywords)
            features['Abnormal_URL'] = 1 if any(kw in url.lower() for kw in self.phishing_keywords) else 0
            
            # Redirects
            features['Redirect'] = 1 if url.count('/') > 3 else 0
            
            # === FEATURES REQUIRING EXTERNAL DATA (Set intelligently based on URL patterns) ===
            
            # Domain registration length - estimate based on domain structure
            tld = self._extract_tld(domain)
            features['Domain_registeration_length'] = 1 if tld in self.legitimate_tlds else 0
            
            # Rest set to 0 as we can't extract without external APIs
            features['Favicon'] = 0
            features['Request_URL'] = 0
            features['URL_of_Anchor'] = 0
            features['Links_in_tags'] = 0
            features['SFH'] = 0  # Server Form Handler
            features['Submitting_to_email'] = 0
            features['on_mouseover'] = 0
            features['RightClick'] = 0
            features['popUpWidnow'] = 0
            features['Iframe'] = 0
            features['age_of_domain'] = 0
            features['DNSRecord'] = 0
            features['web_traffic'] = 0
            features['Page_Rank'] = 0
            features['Google_Index'] = 0
            features['Links_pointing_to_page'] = 0
            features['Statistical_report'] = 0
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return self._default_features()
        
        return features

    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(pattern, domain))

    def _extract_tld(self, domain: str) -> str:
        """Extract top-level domain from domain name"""
        parts = domain.split('.')
        if len(parts) > 0:
            return parts[-1].lower()
        return ''

    def _default_features(self) -> dict:
        """Return default zero features for error handling"""
        return {
            'having_IP_Address': 0,
            'URL_Length': 0,
            'Shortining_Service': 0,
            'having_At_Symbol': 0,
            'double_slash_redirecting': 0,
            'Prefix_Suffix': 0,
            'having_Sub_Domain': 0,
            'SSLfinal_State': 0,
            'Domain_registeration_length': 0,
            'Favicon': 0,
            'port': 0,
            'HTTPS_token': 0,
            'Request_URL': 0,
            'URL_of_Anchor': 0,
            'Links_in_tags': 0,
            'SFH': 0,
            'Submitting_to_email': 0,
            'Abnormal_URL': 0,
            'Redirect': 0,
            'on_mouseover': 0,
            'RightClick': 0,
            'popUpWidnow': 0,
            'Iframe': 0,
            'age_of_domain': 0,
            'DNSRecord': 0,
            'web_traffic': 0,
            'Page_Rank': 0,
            'Google_Index': 0,
            'Links_pointing_to_page': 0,
            'Statistical_report': 0,
        }

    def extract_features_dataframe(self, url: str) -> pd.DataFrame:
        """
        Extract features from URL and return as pandas DataFrame
        This format is suitable for model prediction
        """
        features = self.extract_features(url)
        
        # Ensure features are in correct order matching model's training
        feature_order = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report'
        ]
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([[features.get(col, 0) for col in feature_order]], columns=feature_order)
        return df


# Example usage:
if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://secure-login-bank.xyz",
        "https://amazon.com",
    ]
    
    for url in test_urls:
        print(f"\nURL: {url}")
        phishing_score = extractor.calculate_phishing_score(url)
        print(f"Phishing Score: {phishing_score}/100")
        features_df = extractor.extract_features_dataframe(url)
        print(f"Features: {features_df.shape[1]} columns")


