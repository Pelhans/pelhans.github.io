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

  /* ---------------- 在线编辑（语雀式 + 同步 GitHub） ---------------- */
  function initEditor() {
    var cfgEl = document.getElementById('yqEditConfig');
    if (!cfgEl) return;
    var cfg;
    try { cfg = JSON.parse(cfgEl.textContent); } catch (e) { return; }
    if (!cfg || !cfg.enabled) return;

    var TOKEN_KEY = 'yq-github-token';
    var USER_KEY = 'yq-github-user';
    var siteBase = '';

    var editBtn = document.getElementById('yqEditBtn');
    var newBtn = document.getElementById('yqNewBtn');
    var modal = document.getElementById('yqEditModal');
    var tokenModal = document.getElementById('yqTokenModal');
    if (!editBtn || !modal || !tokenModal) return;

    var els = {
      file: document.getElementById('yqEditFile'),
      text: document.getElementById('yqEditText'),
      preview: document.getElementById('yqEditPreview'),
      status: document.getElementById('yqEditStatus'),
      save: document.getElementById('yqEditSave'),
      user: document.getElementById('yqEditUser'),
      logout: document.getElementById('yqEditLogout'),
      tokenInput: document.getElementById('yqTokenInput'),
      tokenOk: document.getElementById('yqTokenOk'),
      tokenStatus: document.getElementById('yqTokenStatus')
    };

    function api(path, opts) {
      opts = opts || {};
      var token = localStorage.getItem(TOKEN_KEY);
      var headers = { 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch('https://api.github.com' + path, {
        method: opts.method || 'GET',
        headers: headers,
        body: opts.body ? JSON.stringify(opts.body) : undefined
      });
    }

    function setStatus(el, msg, type) {
      el.textContent = msg || '';
      el.className = 'yq-edit-status' + (type ? ' is-' + type : '');
    }

    function allowedUsers() {
      return (cfg.allowed_users || []).map(function (u) { return String(u).toLowerCase(); });
    }

    // 验证 token 并比对白名单
    function verifyToken() {
      var token = localStorage.getItem(TOKEN_KEY);
      if (!token) return Promise.resolve(false);
      return api('/user').then(function (r) {
        if (!r.ok) { throw new Error('token'); }
        return r.json();
      }).then(function (u) {
        localStorage.setItem(USER_KEY, u.login);
        return allowedUsers().indexOf(String(u.login).toLowerCase()) !== -1
          ? u.login : false;
      }).catch(function () { return false; });
    }

    function showEditForUser(login) {
      editBtn.hidden = false;
      if (newBtn) newBtn.hidden = false;
      els.user.textContent = login;
    }

    // 启动时：若已有 token，验证是否在白名单
    verifyToken().then(function (ok) {
      if (ok) showEditForUser(ok);
    });

    /* ---- Token 弹窗 ---- */
    function openToken() {
      els.tokenInput.value = '';
      setStatus(els.tokenStatus, '');
      tokenModal.hidden = false;
      setTimeout(function () { els.tokenInput.focus(); }, 50);
    }
    function closeToken() { tokenModal.hidden = true; }

    els.tokenOk.addEventListener('click', function () {
      var t = els.tokenInput.value.trim();
      if (!t) { setStatus(els.tokenStatus, '请输入 Token', 'err'); return; }
      setStatus(els.tokenStatus, '验证中…');
      localStorage.setItem(TOKEN_KEY, t);
      verifyToken().then(function (ok) {
        if (!ok) {
          localStorage.removeItem(TOKEN_KEY);
          setStatus(els.tokenStatus, 'Token 无效，或该账号不在编辑白名单中', 'err');
          return;
        }
        closeToken();
        showEditForUser(ok);
        openEditor();
      }).catch(function () {
        localStorage.removeItem(TOKEN_KEY);
        setStatus(els.tokenStatus, '验证失败，请重试', 'err');
      });
    });

    els.logout.addEventListener('click', function () {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      editBtn.hidden = true;
      els.user.textContent = '';
      setStatus(els.status, '');
    });

    /* ---- 编辑器 ---- */
    var currentPath = '';
    var currentSha = '';
    var previewTimer = null;

    function openEditor() {
      currentPath = editBtn.getAttribute('data-post-path');
      els.file.textContent = currentPath;
      setStatus(els.status, '正在加载原文…');
      modal.hidden = false;
      els.save.disabled = true;

      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(currentPath) + '?ref=' + cfg.branch)
        .then(function (r) {
          if (!r.ok) throw new Error('load');
          return r.json();
        })
        .then(function (data) {
          currentSha = data.sha;
          var content = decodeURIComponent(escape(window.atob(data.content.replace(/\s/g, ''))));
          els.text.value = content;
          setStatus(els.status, '已加载', 'ok');
          els.save.disabled = false;
          renderPreview();
        })
        .catch(function () {
          setStatus(els.status, '加载原文失败（无权限或网络问题）', 'err');
        });
    }

    function closeEditor() { modal.hidden = true; }

    function renderPreview() {
      var md = els.text.value;
      els.preview.innerHTML = '<div class="yq-preview-empty">渲染中…</div>';
      // 优先使用 GitHub 渲染接口，效果最贴近语雀
      fetch('https://api.github.com/markdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: md, mode: 'gfm', context: cfg.owner + '/' + cfg.repo })
      }).then(function (r) {
        if (!r.ok) throw new Error('preview');
        return r.text();
      }).then(function (html) {
        els.preview.innerHTML = html;
        if (window.MathJax) {
          try { window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub, els.preview]); } catch (e) {}
        }
      }).catch(function () {
        var div = document.createElement('div');
        div.className = 'yq-preview-empty';
        div.textContent = '预览不可用（可能触发 GitHub 频率限制），保存后线上可见效果。';
        els.preview.innerHTML = '';
        els.preview.appendChild(div);
      });
    }

    els.text.addEventListener('input', function () {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(renderPreview, 400);
    });

    els.save.addEventListener('click', function () {
      if (!currentPath || !currentSha) return;
      setStatus(els.status, '保存中…');
      els.save.disabled = true;
      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(currentPath), {
        method: 'PUT',
        body: {
          message: 'docs(' + currentPath + '): update via online editor',
          content: window.btoa(unescape(encodeURIComponent(els.text.value))),
          sha: currentSha,
          branch: cfg.branch
        }
      }).then(function (r) {
        if (!r.ok) throw new Error('save');
        return r.json();
      }).then(function (data) {
        currentSha = data.content.sha;
        setStatus(els.status, '已同步到 GitHub ✓ 稍后站点重新构建即可见', 'ok');
        setTimeout(closeEditor, 1400);
      }).catch(function () {
        setStatus(els.status, '保存失败（权限不足或冲突，请重试）', 'err');
        els.save.disabled = false;
      });
    });

    /* ---- 打开流程：有 token 直接开，无则先要 token ---- */
    editBtn.addEventListener('click', function () {
      verifyToken().then(function (ok) {
        if (!ok) { openToken(); return; }
        showEditForUser(ok);
        openEditor();
      });
    });

    // 关闭事件（背景 / ✕ / 取消）
    Array.prototype.forEach.call(modal.querySelectorAll('[data-close]'), function (el) {
      el.addEventListener('click', closeEditor);
    });
    Array.prototype.forEach.call(tokenModal.querySelectorAll('[data-token-close]'), function (el) {
      el.addEventListener('click', closeToken);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeEditor(); closeToken(); }
    });

    /* ============ 新建文档 ============ */
    var newModal = document.getElementById('yqNewModal');
    var delModal = document.getElementById('yqDelModal');
    var booksData = [];
    try {
      var bd = document.getElementById('yqBooksData');
      if (bd) booksData = JSON.parse(bd.textContent);
    } catch (e) {}

    var slugify = function (s) {
      return String(s).trim().toLowerCase()
        .replace(/[^\w\u4e00-\u9fa5]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'untitled';
    };

    function fillBookOptions() {
      var sel = document.getElementById('yqNewBook');
      var gsel = document.getElementById('yqNewGroup');
      sel.innerHTML = '';
      gsel.innerHTML = '';
      booksData.forEach(function (b, i) {
        var o = document.createElement('option');
        o.value = i;
        o.textContent = (b.icon ? b.icon + ' ' : '') + b.title;
        sel.appendChild(o);
      });
      fillGroupOptions();
      sel.onchange = fillGroupOptions;
    }
    function fillGroupOptions() {
      var sel = document.getElementById('yqNewBook');
      var gsel = document.getElementById('yqNewGroup');
      var b = booksData[sel.value];
      gsel.innerHTML = '';
      if (!b) return;
      (b.groups || []).forEach(function (g, i) {
        var o = document.createElement('option');
        o.value = i;
        o.textContent = g.title;
        gsel.appendChild(o);
      });
    }

    function openNew() {
      document.getElementById('yqNewTitle').value = '';
      document.getElementById('yqNewText').value = '';
      setStatus(document.getElementById('yqNewStatus'), '');
      fillBookOptions();
      newModal.hidden = false;
      setTimeout(function () { document.getElementById('yqNewTitle').focus(); }, 50);
    }
    function closeNew() { newModal.hidden = true; }

    document.getElementById('yqNewSave').addEventListener('click', function () {
      var titleEl = document.getElementById('yqNewTitle');
      var textEl = document.getElementById('yqNewText');
      var statusEl = document.getElementById('yqNewStatus');
      var title = titleEl.value.trim();
      if (!title) { setStatus(statusEl, '请填写标题', 'err'); return; }

      var b = booksData[document.getElementById('yqNewBook').value];
      var g = (b.groups || [])[document.getElementById('yqNewGroup').value];
      var prefix = (g && g.prefix && g.prefix[0]) ? g.prefix[0] + '_' : '';
      var tag = (g && g.tag && g.tag[0]) ? g.tag[0] : b.title;
      var date = new Date();
      var ymd = date.getFullYear() + '-' +
        ('0' + (date.getMonth() + 1)).slice(-2) + '-' +
        ('0' + date.getDate()).slice(-2);
      var slug = prefix + slugify(title);
      var path = '_posts/' + ymd + '-' + slug + '.markdown';

      var body = '---\n' +
        'layout: post\n' +
        'title: "' + title.replace(/"/g, '\\"') + '"\n' +
        'subtitle: ""\n' +
        'date: ' + ymd + ' ' +
        ('0' + date.getHours()).slice(-2) + ':' +
        ('0' + date.getMinutes()).slice(-2) + ':' +
        ('0' + date.getSeconds()).slice(-2) + '\n' +
        'author: "' + (localStorage.getItem(USER_KEY) || cfg.owner) + '"\n' +
        'tags: [' + tag + ']\n' +
        '---\n\n' + textEl.value + '\n';

      setStatus(statusEl, '创建中…');
      this.disabled = true;
      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(path), {
        method: 'POST',
        body: {
          message: 'docs: new post ' + path,
          content: window.btoa(unescape(encodeURIComponent(body))),
          branch: cfg.branch
        }
      }).then(function (r) {
        if (!r.ok) throw new Error('create');
        return r.json();
      }).then(function (data) {
        setStatus(statusEl, '已创建 ✓ 正在跳转…', 'ok');
        setTimeout(function () {
          window.location.href = (siteBase || '') + '/' + data.content.path
            .replace(/^_posts\//, '')
            .replace(/\.markdown$/, '')
            .replace(/(\d{4})-(\d{2})-(\d{2})-(.*)/, '$1/$2/$3/$4/');
        }, 1000);
      }).catch(function () {
        setStatus(statusEl, '创建失败（文件名可能已存在或权限不足）', 'err');
        this.disabled = false;
      }.bind(this));
    });

    if (newBtn) {
      newBtn.addEventListener('click', function () {
        verifyToken().then(function (ok) {
          if (!ok) { openToken(); return; }
          showEditForUser(ok);
          openNew();
        });
      });
    }
    Array.prototype.forEach.call(newModal.querySelectorAll('[data-new-close]'), function (el) {
      el.addEventListener('click', closeNew);
    });

    /* ============ 删除文档 ============ */
    var delPath = '';
    var delName = '';
    function openDelete() {
      delPath = editBtn.getAttribute('data-post-path');
      delName = delPath.split('/').pop();
      document.getElementById('yqDelName').textContent = delName;
      setStatus(document.getElementById('yqDelStatus'), '');
      delModal.hidden = false;
    }
    function closeDelete() { delModal.hidden = true; }

    document.getElementById('yqEditDelete').addEventListener('click', function () {
      closeEditor();
      openDelete();
    });
    document.getElementById('yqDelOk').addEventListener('click', function () {
      var statusEl = document.getElementById('yqDelStatus');
      setStatus(statusEl, '删除中…');
      this.disabled = true;
      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(delPath) + '?ref=' + cfg.branch)
        .then(function (r) { if (!r.ok) throw new Error('get'); return r.json(); })
        .then(function (data) {
          return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(delPath), {
            method: 'DELETE',
            body: {
              message: 'docs: delete ' + delPath,
              sha: data.sha,
              branch: cfg.branch
            }
          });
        })
        .then(function (r) { if (!r.ok) throw new Error('del'); return r; })
        .then(function () {
          setStatus(statusEl, '已删除 ✓ 即将返回首页', 'ok');
          setTimeout(function () { window.location.href = (siteBase || '') + '/'; }, 1000);
        })
        .catch(function () {
          setStatus(statusEl, '删除失败（权限不足或文件已被改动）', 'err');
          this.disabled = false;
        }.bind(this));
    });
    Array.prototype.forEach.call(delModal.querySelectorAll('[data-del-close]'), function (el) {
      el.addEventListener('click', closeDelete);
    });
  }

  ready(function () {
    initTree();
    initSidebarToggle();
    initOutline();
    initSearch();
    initEditor();
  });
})();
