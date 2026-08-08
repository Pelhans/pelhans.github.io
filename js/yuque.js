(function () {
  'use strict';

  /* ---------------- 侧边栏折叠 / 移动端抽屉 ---------------- */
  var STORE_KEY = 'yq-open-groups';

  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); }
    catch (e) { return {}; }
  }
  function writeStore(v) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(v)); } catch (e) {}
  }

  function initTree() {
    var store = readStore();
    var groups = document.querySelectorAll('.yq-tree-group');

    Array.prototype.forEach.call(groups, function (g) {
      var nameEl = g.querySelector('.yq-tree-group-name');
      var key = nameEl ? nameEl.textContent.trim() : '';
      // 当前文档所在分组始终展开；否则读取本地记忆
      if (!g.classList.contains('is-open') && store[key]) {
        g.classList.add('is-open');
      }

      var title = g.querySelector('.yq-tree-group-title');
      if (!title) return;
      var toggle = function () {
        g.classList.toggle('is-open');
        store[key] = g.classList.contains('is-open');
        writeStore(store);
      };
      title.addEventListener('click', toggle);
      title.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
    });

    // 滚动到激活文档
    var active = document.querySelector('.yq-tree-doc.is-active');
    var sidebar = document.querySelector('.yq-sidebar');
    if (active && sidebar) {
      var offset = active.offsetTop - sidebar.clientHeight / 2;
      if (offset > 0) sidebar.scrollTop = offset;
    }
  }

  function initSidebarToggle() {
    var btn = document.getElementById('yqSidebarToggle');
    var sidebar = document.querySelector('.yq-sidebar');
    if (!btn || !sidebar) {
      if (btn) btn.style.display = 'none';
      return;
    }
    var mask = document.createElement('div');
    mask.className = 'yq-mask';
    document.body.appendChild(mask);

    function close() { sidebar.classList.remove('is-open'); mask.classList.remove('is-open'); }
    btn.addEventListener('click', function () {
      sidebar.classList.toggle('is-open');
      mask.classList.toggle('is-open');
    });
    mask.addEventListener('click', close);
  }

  /* ---------------- 右侧大纲 ---------------- */
  function initOutline() {
    var outline = document.getElementById('yqOutline');
    var content = document.querySelector('.yq-content');
    if (!outline || !content) return;

    var heads = content.querySelectorAll('h1, h2, h3');
    var items = [];
    var used = {};

    Array.prototype.forEach.call(heads, function (h) {
      var text = (h.textContent || '').trim();
      if (!text) return;
      if (!h.id) {
        var slug = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-+|-+$/g, '');
        if (!slug) slug = 'section';
        if (used[slug]) { used[slug]++; slug = slug + '-' + used[slug]; } else { used[slug] = 1; }
        h.id = slug;
      }
      items.push({ id: h.id, text: text, level: parseInt(h.tagName.substring(1), 10), el: h });
    });

    if (items.length < 2) { outline.style.display = 'none'; return; }

    var minLevel = items.reduce(function (m, i) { return Math.min(m, i.level); }, 6);
    var html = '<div class="yq-outline-title">本文目录</div><ul>';
    items.forEach(function (i) {
      var lv = i.level - minLevel + 2; // 2,3,4
      if (lv > 4) lv = 4;
      html += '<li><a class="lv-' + lv + '" href="#' + i.id + '" data-target="' + i.id + '">' +
        i.text.replace(/</g, '&lt;') + '</a></li>';
    });
    html += '</ul>';
    outline.innerHTML = html;

    var links = outline.querySelectorAll('a');

    links = Array.prototype.slice.call(links);
    links.forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var target = document.getElementById(a.getAttribute('data-target'));
        if (!target) return;
        var top = target.getBoundingClientRect().top + window.pageYOffset - 72;
        window.scrollTo({ top: top, behavior: 'smooth' });
        history.replaceState(null, '', '#' + a.getAttribute('data-target'));
      });
    });

    var ticking = false;
    function spy() {
      var pos = window.pageYOffset + 100;
      var currentIndex = 0;
      for (var i = 0; i < items.length; i++) {
        if (items[i].el.offsetTop <= pos) currentIndex = i; else break;
      }
      links.forEach(function (a, idx) {
        if (idx === currentIndex) a.classList.add('is-active');
        else a.classList.remove('is-active');
      });
      ticking = false;
    }
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(spy); }
    });
    spy();
  }

  /* ---------------- 搜索 ---------------- */
  function initSearch() {
    var input = document.getElementById('yqSearchInput');
    var box = document.getElementById('yqSearchResult');
    if (!input || !box) return;

    var data = window.YQ_DOCS || [];

    function render(list, kw) {
      if (!list.length) {
        box.innerHTML = '<div class="yq-search-empty">没有找到相关文档</div>';
      } else {
        box.innerHTML = list.slice(0, 12).map(function (d) {
          return '<a href="' + d.url + '">' +
            '<div class="yq-sr-title">' + highlight(d.title, kw) + '</div>' +
            '<div class="yq-sr-book">' + d.book + (d.sub ? ' · ' + d.sub : '') + '</div>' +
            '</a>';
        }).join('');
      }
      box.classList.add('is-open');
    }

    function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
    function highlight(text, kw) {
      var t = esc(text);
      if (!kw) return t;
      var i = t.toLowerCase().indexOf(kw.toLowerCase());
      if (i < 0) return t;
      return t.slice(0, i) + '<mark>' + t.slice(i, i + kw.length) + '</mark>' + t.slice(i + kw.length);
    }

    input.addEventListener('input', function () {
      var kw = input.value.trim().toLowerCase();
      if (!kw) { box.classList.remove('is-open'); return; }
      var list = data.filter(function (d) {
        return (d.title + ' ' + (d.sub || '') + ' ' + d.book + ' ' + (d.tags || '')).toLowerCase().indexOf(kw) >= 0;
      });
      render(list, input.value.trim());
    });

    input.addEventListener('focus', function () {
      if (input.value.trim()) box.classList.add('is-open');
    });

    document.addEventListener('click', function (e) {
      if (!box.contains(e.target) && e.target !== input) box.classList.remove('is-open');
    });

    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); input.focus(); }
      if (e.key === 'Escape') { box.classList.remove('is-open'); input.blur(); }
    });
  }

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    initTree();
    initSidebarToggle();
    initOutline();
    initSearch();
  });
})();
