"""
Enhanced Image Analyzer with detailed category detection
Расширенный анализатор изображений с детальным определением категорий
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ka_modules.image_analyzer import SmartImageAnalyzer
from ka_modules.analysis_categories import AnalysisCategories, find_category_for_keyword

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnalysisConfig:
    """Configuration for enhanced analysis"""
    enable_fantasy_detection: bool = True
    enable_material_detection: bool = True
    enable_fine_details: bool = True
    enable_lighting_analysis: bool = True
    enable_style_detection: bool = True
    min_confidence: float = 0.5
    debug_mode: bool = False


class EnhancedAnalyzer(SmartImageAnalyzer):
    """
    Enhanced analyzer that extends SmartImageAnalyzer
    Maintains backward compatibility while adding detailed analysis
    """
    
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.categories = AnalysisCategories()
        self.enhanced_mode = False  # По умолчанию выключен для безопасности
        self.config = EnhancedAnalysisConfig()
        
        # Кэш для результатов анализа
        self._analysis_cache = {}
        
        logger.info("EnhancedAnalyzer initialized with enhanced_mode=False")
    
    def analyze(self, image, task_type=None, enhanced=None):
        """
        Analyze image with optional enhanced mode
        
        Args:
            image: PIL Image to analyze
            task_type: Type of task (for compatibility)
            enhanced: Override enhanced mode (True/False/None)
        
        Returns:
            dict: Analysis results with optional enhanced_analysis field
        """
        # Определяем, использовать ли расширенный режим
        use_enhanced = enhanced if enhanced is not None else self.enhanced_mode
        
        # Всегда выполняем базовый анализ для совместимости
        start_time = time.time()
        base_result = super().analyze(image, task_type)
        base_time = time.time() - start_time
        
        # Добавляем время анализа
        base_result['analysis_time'] = base_time
        
        if not use_enhanced:
            return base_result
        
        # Расширенный анализ
        try:
            enhanced_start = time.time()
            enhanced_result = self._perform_enhanced_analysis(image, base_result)
            enhanced_time = time.time() - enhanced_start
            
            # Добавляем расширенные результаты
            base_result['enhanced_analysis'] = enhanced_result
            base_result['enhanced_analysis']['analysis_time'] = enhanced_time
            base_result['total_analysis_time'] = base_time + enhanced_time
            
            logger.info(f"Enhanced analysis completed in {enhanced_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}", exc_info=self.config.debug_mode)
            # Возвращаем базовый результат без падения
            base_result['enhanced_analysis_error'] = str(e)
        
        return base_result
    
    def _perform_enhanced_analysis(self, image, base_result) -> Dict[str, Any]:
        """
        Perform detailed enhanced analysis
        
        Args:
            image: PIL Image
            base_result: Results from base analysis
        
        Returns:
            dict: Enhanced analysis results
        """
        enhanced = {
            'fantasy_elements': [],
            'materials': [],
            'fine_details': [],
            'lighting': {},
            'style': {},
            'confidence_scores': {},
            'detected_categories': []
        }
        
        # Извлекаем текстовые описания для анализа
        descriptions = self._extract_descriptions(base_result)
        
        # 1. Анализ фэнтези элементов
        if self.config.enable_fantasy_detection:
            enhanced['fantasy_elements'] = self._detect_fantasy_elements(descriptions)
            if enhanced['fantasy_elements']:
                enhanced['detected_categories'].append('fantasy')
        
        # 2. Определение материалов
        if self.config.enable_material_detection:
            enhanced['materials'] = self._detect_materials(descriptions)
            if enhanced['materials']:
                enhanced['detected_categories'].append('materials')
        
        # 3. Мелкие детали
        if self.config.enable_fine_details:
            enhanced['fine_details'] = self._detect_fine_details(descriptions)
            if enhanced['fine_details']:
                enhanced['detected_categories'].append('details')
        
        # 4. Анализ освещения
        if self.config.enable_lighting_analysis:
            enhanced['lighting'] = self._analyze_lighting(descriptions)
            if enhanced['lighting']:
                enhanced['detected_categories'].append('lighting')
        
        # 5. Определение стиля
        if self.config.enable_style_detection:
            enhanced['style'] = self._analyze_style(descriptions)
            if enhanced['style']:
                enhanced['detected_categories'].append('style')
        
        # Общая статистика
        enhanced['total_elements_found'] = (
            len(enhanced['fantasy_elements']) +
            len(enhanced['materials']) +
            len(enhanced['fine_details'])
        )
        
        return enhanced
    
    def _extract_descriptions(self, base_result) -> Dict[str, str]:
        """Extract all text descriptions from base analysis"""
        descriptions = {
            'main': str(base_result.get('description', '')).lower()
        }
        
        # Обработка objects - это словарь с подразделами
        objects_data = base_result.get('objects', {})
        if isinstance(objects_data, dict):
            # Извлекаем все объекты из разных категорий
            all_objects = []
            
            # main objects
            if 'main' in objects_data and isinstance(objects_data['main'], list):
                all_objects.extend(objects_data['main'])
            
            # secondary objects
            if 'secondary' in objects_data and isinstance(objects_data['secondary'], list):
                all_objects.extend(objects_data['secondary'])
            
            # all objects (может содержать дубликаты, но это ok для поиска)
            if 'all' in objects_data and isinstance(objects_data['all'], list):
                all_objects.extend(objects_data['all'])
            
            descriptions['objects'] = ' '.join(all_objects).lower()
            
            # Добавляем информацию о количестве объектов
            if 'counts' in objects_data and isinstance(objects_data['counts'], dict):
                count_info = []
                for obj, count in objects_data['counts'].items():
                    if count > 1:
                        count_info.append(f"{count} {obj}s")
                    else:
                        count_info.append(f"{count} {obj}")
                descriptions['object_counts'] = ' '.join(count_info).lower()
        
        # Обработка style - это словарь
        style_data = base_result.get('style', {})
        if isinstance(style_data, dict):
            style_parts = []
            for key, value in style_data.items():
                if value:
                    style_parts.append(str(value))
            descriptions['style'] = ' '.join(style_parts).lower()
        else:
            descriptions['style'] = str(style_data).lower()
        
        # Обработка environment - тоже словарь
        env_data = base_result.get('environment', {})
        if isinstance(env_data, dict):
            env_parts = []
            for key, value in env_data.items():
                if value:
                    env_parts.append(str(value))
            descriptions['environment'] = ' '.join(env_parts).lower()
        else:
            descriptions['environment'] = str(env_data).lower()
        
        # Обработка composition
        comp_data = base_result.get('composition', {})
        if isinstance(comp_data, dict):
            comp_parts = []
            for key, value in comp_data.items():
                if value and not isinstance(value, (int, float)):
                    comp_parts.append(str(value))
            descriptions['composition'] = ' '.join(comp_parts).lower()
        
        # Объединенное описание для полного поиска
        non_empty_descriptions = [v for v in descriptions.values() if v and v.strip()]
        descriptions['combined'] = ' '.join(non_empty_descriptions)
        
        return descriptions
    
    def _detect_fantasy_elements(self, descriptions: Dict[str, str]) -> List[Dict]:
        """Detect fantasy and fantastical elements"""
        found_elements = []
        combined_text = descriptions['combined']
        
        # Проходим по всем категориям фэнтези элементов
        for category, subcategories in self.categories.FANTASY_ELEMENTS.items():
            for subcat, keywords in subcategories.items():
                if isinstance(keywords, dict):
                    # Вложенная структура (например, medieval_fantasy)
                    for item_type, item_keywords in keywords.items():
                        for keyword in item_keywords:
                            if keyword.lower() in combined_text:
                                element = {
                                    'category': category,
                                    'subcategory': subcat,
                                    'type': item_type,
                                    'detected': keyword,
                                    'confidence': self._calculate_confidence(keyword, combined_text)
                                }
                                if element['confidence'] >= self.config.min_confidence:
                                    found_elements.append(element)
                else:
                    # Простой список ключевых слов
                    for keyword in keywords:
                        if keyword.lower() in combined_text:
                            element = {
                                'category': category,
                                'subcategory': subcat,
                                'detected': keyword,
                                'confidence': self._calculate_confidence(keyword, combined_text)
                            }
                            if element['confidence'] >= self.config.min_confidence:
                                found_elements.append(element)
        
        # Сортируем по уверенности
        found_elements.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Группируем по категориям для удобства
        return self._group_by_category(found_elements)
    
    def _detect_materials(self, descriptions: Dict[str, str]) -> List[Dict]:
        """Detect materials and textures"""
        found_materials = []
        combined_text = descriptions['combined']
        
        for material_type, subcategories in self.categories.MATERIALS.items():
            for subcat, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        material = {
                            'type': material_type,
                            'subcategory': subcat,
                            'material': keyword,
                            'confidence': self._calculate_confidence(keyword, combined_text)
                        }
                        if material['confidence'] >= self.config.min_confidence:
                            found_materials.append(material)
        
        # Удаляем дубликаты и сортируем
        unique_materials = self._remove_duplicates(found_materials, key='material')
        return sorted(unique_materials, key=lambda x: x['confidence'], reverse=True)
    
    def _detect_fine_details(self, descriptions: Dict[str, str]) -> List[Dict]:
        """Detect fine details like jewelry, tattoos, etc."""
        found_details = []
        combined_text = descriptions['combined']
        
        for detail_type, subcategories in self.categories.FINE_DETAILS.items():
            for subcat, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        detail = {
                            'type': detail_type,
                            'subcategory': subcat,
                            'detail': keyword,
                            'confidence': self._calculate_confidence(keyword, combined_text)
                        }
                        if detail['confidence'] >= self.config.min_confidence:
                            found_details.append(detail)
        
        return sorted(found_details, key=lambda x: x['confidence'], reverse=True)
    
    def _analyze_lighting(self, descriptions: Dict[str, str]) -> Dict[str, Any]:
        """Analyze lighting and atmosphere"""
        lighting_info = {
            'time_of_day': None,
            'light_quality': [],
            'atmosphere': [],
            'mood': None
        }
        
        combined_text = descriptions['combined']
        
        # Время суток
        for time_period, keywords in self.categories.LIGHTING_ATMOSPHERE['time_of_day'].items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    lighting_info['time_of_day'] = time_period
                    break
            if lighting_info['time_of_day']:
                break
        
        # Качество света
        for quality_type, keywords in self.categories.LIGHTING_ATMOSPHERE['light_quality'].items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    lighting_info['light_quality'].append({
                        'type': quality_type,
                        'detected': keyword
                    })
        
        # Атмосфера
        for atmosphere_type, keywords in self.categories.LIGHTING_ATMOSPHERE['atmosphere'].items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    if atmosphere_type == 'mood' and not lighting_info['mood']:
                        lighting_info['mood'] = keyword
                    else:
                        lighting_info['atmosphere'].append({
                            'type': atmosphere_type,
                            'detected': keyword
                        })
        
        return lighting_info
    
    def _analyze_style(self, descriptions: Dict[str, str]) -> Dict[str, Any]:
        """Analyze artistic style"""
        style_info = {
            'primary_style': None,
            'style_elements': [],
            'techniques': [],
            'cultural_influence': None
        }
        
        combined_text = descriptions['combined']
        
        # Проходим по всем стилистическим категориям
        for style_category, subcategories in self.categories.ARTISTIC_STYLES.items():
            for subcat, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        element = {
                            'category': style_category,
                            'subcategory': subcat,
                            'style': keyword,
                            'confidence': self._calculate_confidence(keyword, combined_text)
                        }
                        
                        # Определяем основной стиль
                        if not style_info['primary_style'] and element['confidence'] > 0.7:
                            style_info['primary_style'] = keyword
                        
                        # Добавляем в элементы стиля
                        style_info['style_elements'].append(element)
                        
                        # Определяем культурное влияние
                        if style_category == 'cultural_styles' and not style_info['cultural_influence']:
                            style_info['cultural_influence'] = subcat
                        
                        # Техники
                        if style_category == 'technique':
                            style_info['techniques'].append(keyword)
        
        # Сортируем элементы по уверенности
        style_info['style_elements'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return style_info
    
    def _calculate_confidence(self, keyword: str, text: str) -> float:
        """
        Calculate confidence score for keyword detection
        
        Simple implementation - can be enhanced with:
        - Context analysis
        - Proximity to other related keywords
        - Frequency of occurrence
        """
        # Базовая уверенность
        confidence = 0.6
        
        # Повышаем уверенность, если ключевое слово встречается несколько раз
        count = text.lower().count(keyword.lower())
        if count > 1:
            confidence += 0.1 * min(count - 1, 3)  # До +0.3
        
        # Повышаем уверенность для более длинных и специфичных ключевых слов
        if len(keyword) > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _group_by_category(self, elements: List[Dict]) -> List[Dict]:
        """Group elements by category for better organization"""
        grouped = {}
        
        for element in elements:
            category = element.get('category', 'unknown')
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(element)
        
        # Преобразуем в список с категориями
        result = []
        for category, items in grouped.items():
            result.append({
                'category': category,
                'count': len(items),
                'elements': items[:10]  # Ограничиваем количество элементов
            })
        
        return result
    
    def _remove_duplicates(self, items: List[Dict], key: str) -> List[Dict]:
        """Remove duplicate items based on key"""
        seen = set()
        unique = []
        
        for item in items:
            if item[key] not in seen:
                seen.add(item[key])
                unique.append(item)
        
        return unique
    
    def set_enhanced_mode(self, enabled: bool):
        """Enable or disable enhanced mode"""
        self.enhanced_mode = enabled
        logger.info(f"Enhanced mode set to: {enabled}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about analyzer performance"""
        stats = {
            'enhanced_mode': self.enhanced_mode,
            'total_keywords': len(AnalysisCategories.get_all_keywords()),
            'categories': {
                'fantasy': len(AnalysisCategories.get_all_keywords('FANTASY_ELEMENTS')),
                'materials': len(AnalysisCategories.get_all_keywords('MATERIALS')),
                'details': len(AnalysisCategories.get_all_keywords('FINE_DETAILS')),
                'lighting': len(AnalysisCategories.get_all_keywords('LIGHTING_ATMOSPHERE')),
                'style': len(AnalysisCategories.get_all_keywords('ARTISTIC_STYLES'))
            }
        }
        return stats