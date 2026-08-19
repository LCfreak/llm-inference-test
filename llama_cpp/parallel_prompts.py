"""
Parallel prompt set for cross-lingual tokenization-premium benchmarking.

Design notes (important for research validity):
- 15 sentences per language, SAME semantic content across languages
  (translation-matched, not independently sampled) so that token-count
  differences reflect tokenizer/script behavior, not topic or length choice.
- Sentences are bucketed into short / medium / long (5 each) by approximate
  English word count, so you can also report results *within* length bucket
  and check the overhead ratio isn't just an artifact of one bucket.
- Hindi and Nepali translations below are a reasonable-effort parallel set.
  For a paper/report you should have a native speaker (or two, for IAA)
  review/correct these before treating results as citable — MT-quality
  parallel text is a common confound in tokenization-fairness papers.
- Keep punctuation/spacing conventions consistent per language (e.g. Hindi/
  Nepali use the danda "।" as a sentence-final marker) since tokenizers are
  sensitive to punctuation too.
"""

PROMPTS = {
    "English": [
        # short (5)
        "Explain machine learning briefly.",
        "What causes seasons to change?",
        "Describe how vaccines work.",
        "Why is the sky blue?",
        "Summarize the water cycle.",
        # medium (5)
        "Explain how photosynthesis converts sunlight into chemical energy in plants.",
        "Describe the main differences between a virus and a bacterium.",
        "Explain why compound interest grows faster than simple interest over time.",
        "Describe how earthquakes are caused by movement of tectonic plates.",
        "Explain the basic principle behind how a refrigerator keeps food cold.",
        # long (5)
        "Explain, in simple terms, how neural networks learn from labeled examples by adjusting their internal weights during training.",
        "Describe the process by which rainwater evaporates, forms clouds, and eventually falls back to earth as precipitation.",
        "Explain why countries with different climates have developed very different traditional methods of farming and food storage.",
        "Describe how the human immune system identifies foreign pathogens and produces antibodies to fight future infections.",
        "Explain how renewable energy sources like solar and wind power can reduce a country's dependence on fossil fuels.",
    ],
    "Hindi": [
        # short (5)
        "मशीन लर्निंग को संक्षेप में समझाइए।",
        "मौसम क्यों बदलते हैं?",
        "टीके कैसे काम करते हैं, बताइए।",
        "आकाश नीला क्यों दिखता है?",
        "जल चक्र को संक्षेप में समझाइए।",
        # medium (5)
        "समझाइए कि पौधे प्रकाश संश्लेषण के द्वारा सूर्य के प्रकाश को रासायनिक ऊर्जा में कैसे बदलते हैं।",
        "वायरस और बैक्टीरिया के बीच मुख्य अंतर बताइए।",
        "समझाइए कि समय के साथ चक्रवृद्धि ब्याज साधारण ब्याज से तेज़ी से क्यों बढ़ता है।",
        "बताइए कि टेक्टोनिक प्लेटों की गति से भूकंप कैसे आते हैं।",
        "समझाइए कि रेफ्रिजरेटर भोजन को ठंडा रखने के लिए किस सिद्धांत पर काम करता है।",
        # long (5)
        "सरल शब्दों में समझाइए कि न्यूरल नेटवर्क प्रशिक्षण के दौरान अपने आंतरिक भार को समायोजित करके लेबल किए गए उदाहरणों से कैसे सीखते हैं।",
        "उस प्रक्रिया का वर्णन कीजिए जिसमें वर्षा का जल वाष्पित होकर बादल बनाता है और अंततः वर्षा के रूप में पृथ्वी पर वापस गिरता है।",
        "समझाइए कि अलग-अलग जलवायु वाले देशों ने खेती और भोजन भंडारण की बहुत भिन्न पारंपरिक विधियाँ क्यों विकसित की हैं।",
        "बताइए कि मानव प्रतिरक्षा प्रणाली बाहरी रोगजनकों की पहचान कैसे करती है और भविष्य के संक्रमणों से लड़ने के लिए एंटीबॉडी कैसे बनाती है।",
        "समझाइए कि सौर और पवन ऊर्जा जैसे नवीकरणीय ऊर्जा स्रोत किसी देश की जीवाश्म ईंधन पर निर्भरता को कैसे कम कर सकते हैं।",
    ],
    "Nepali": [
        # short (5)
        "मेसिन लर्निङलाई संक्षेपमा बुझाउनुहोस्।",
        "ऋतुहरू किन परिवर्तन हुन्छन्?",
        "खोपले कसरी काम गर्छ, बताउनुहोस्।",
        "आकाश किन निलो देखिन्छ?",
        "जल चक्रलाई संक्षेपमा बुझाउनुहोस्।",
        # medium (5)
        "बिरुवाहरूले प्रकाश संश्लेषणद्वारा सूर्यको प्रकाशलाई रासायनिक ऊर्जामा कसरी परिवर्तन गर्छन् भनी बुझाउनुहोस्।",
        "भाइरस र ब्याक्टेरियाबीचको मुख्य भिन्नता बताउनुहोस्।",
        "समयसँगै चक्रवृद्धि ब्याज साधारण ब्याजभन्दा किन छिटो बढ्छ भनी बुझाउनुहोस्।",
        "टेक्टोनिक प्लेटहरूको चालले भूकम्प कसरी ल्याउँछ भनी बताउनुहोस्।",
        "रेफ्रिजरेटरले खानालाई चिसो राख्न कुन सिद्धान्तमा काम गर्छ भनी बुझाउनुहोस्।",
        # long (5)
        "सरल भाषामा बुझाउनुहोस् कि न्युरल नेटवर्कहरूले तालिमको क्रममा आफ्नो आन्तरिक तौल समायोजन गरेर लेबल गरिएका उदाहरणहरूबाट कसरी सिक्छन्।",
        "वर्षाको पानी वाष्पीकरण भई बादल बन्ने र अन्ततः वर्षाको रूपमा पृथ्वीमा फर्केर झर्ने प्रक्रियाको वर्णन गर्नुहोस्।",
        "फरक-फरक जलवायु भएका देशहरूले खेती र खाद्य भण्डारणका निकै फरक परम्परागत विधिहरू किन विकास गरे भनी बुझाउनुहोस्।",
        "मानव प्रतिरक्षा प्रणालीले बाहिरी रोगजनकहरू कसरी पहिचान गर्छ र भविष्यका सङ्क्रमणहरूसँग लड्न एन्टिबडी कसरी बनाउँछ भनी बताउनुहोस्।",
        "सौर्य र वायु ऊर्जा जस्ता नवीकरणीय ऊर्जा स्रोतहरूले कुनै देशको जीवाश्म इन्धनमाथिको निर्भरता कसरी घटाउन सक्छन् भनी बुझाउनुहोस्।",
    ],
}

# Length bucket labels aligned by index (0-4 short, 5-9 medium, 10-14 long)
LENGTH_BUCKETS = ["short"] * 5 + ["medium"] * 5 + ["long"] * 5

if __name__ == "__main__":
    for lang, prompts in PROMPTS.items():
        assert len(prompts) == 15, f"{lang} has {len(prompts)} prompts, expected 15"
    print("OK: 15 parallel prompts per language (45 total), buckets aligned.")