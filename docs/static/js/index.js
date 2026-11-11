// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

    // Initialize optional template widgets only when their libraries are present.
    if (typeof bulmaCarousel !== 'undefined') {
        bulmaCarousel.attach('.carousel', options);
    }

    if (typeof bulmaSlider !== 'undefined') {
        bulmaSlider.attach();
    }
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

});

// Project-page section navigation and restrained scroll ambience.
document.addEventListener('DOMContentLoaded', function() {
    const exploreMenu = document.querySelector('.explore-menu');
    if (exploreMenu) {
        exploreMenu.querySelectorAll('nav a').forEach(link => {
            link.addEventListener('click', () => { exploreMenu.open = false; });
        });
        document.addEventListener('click', event => {
            if (!exploreMenu.contains(event.target)) exploreMenu.open = false;
        });
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape') exploreMenu.open = false;
        });
    }

    const railLinks = Array.from(document.querySelectorAll('.section-rail a'));
    const observedSections = railLinks
        .map(link => document.querySelector(link.getAttribute('href')))
        .filter(Boolean);

    if ('IntersectionObserver' in window && observedSections.length) {
        const sectionObserver = new IntersectionObserver(function(entries) {
            const visible = entries
                .filter(entry => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
            if (!visible) return;
            railLinks.forEach(link => {
                link.classList.toggle('is-active', link.getAttribute('href') === '#' + visible.target.id);
            });
        }, { rootMargin: '-25% 0px -58% 0px', threshold: [0.05, 0.2, 0.5] });
        observedSections.forEach(section => sectionObserver.observe(section));
    }

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const root = document.documentElement;
    if (root.classList.contains('intro-pending')) {
        const realLogo = document.querySelector('.intro-logo-stage .pimpc-logo');
        const fontReady = document.fonts && document.fonts.ready
            ? document.fonts.ready
            : Promise.resolve();
        const logoReady = realLogo && realLogo.decode
            ? realLogo.decode().catch(function() {})
            : Promise.resolve();

        Promise.all([fontReady, logoReady]).then(function() {
            window.setTimeout(function() {
                if (realLogo) {
                    const logoRect = realLogo.getBoundingClientRect();
                    const maxScale = Math.max(
                        window.innerWidth / Math.max(logoRect.width, 1),
                        window.innerHeight / Math.max(logoRect.height, 1)
                    ) * 1.06;
                    root.style.setProperty('--intro-logo-x', `${logoRect.left + logoRect.width / 2}px`);
                    root.style.setProperty('--intro-logo-y', `${logoRect.top + logoRect.height / 2}px`);
                    root.style.setProperty('--intro-logo-width', `${logoRect.width}px`);
                    root.style.setProperty('--intro-logo-height', `${logoRect.height}px`);
                    root.style.setProperty('--intro-logo-max-scale', maxScale.toFixed(3));
                }
                window.requestAnimationFrame(function() {
                    root.classList.add('intro-ready');
                });
                window.setTimeout(function() {
                    window.clearTimeout(window.__pimpcIntroFallback);
                    root.classList.add('intro-complete');
                    root.classList.remove('intro-pending');
                    root.classList.remove('intro-ready');
                    try {
                        window.sessionStorage.setItem('pimpc-intro-seen-v10', '1');
                    } catch (error) {
                        // The animation still completes when storage is unavailable.
                    }
                }, 2150);
            }, 700);
        });
    }

    const revealBlocks = Array.from(document.querySelectorAll(
        '.section:not(#top) .method-kicker, .section:not(#top) h2, ' +
        '.section:not(#top) .method-lead, .section:not(#top) .section-subtitle, ' +
        '.section:not(#top) .publication-body > p, .method-equation-card, ' +
        '.bottleneck-grid article, .split-system > div, ' +
        '.interactive-method-figure, .pimpc-player, .iteration-flow article, .consensus-panel, ' +
        '.highlights-intro, .highlight-item, .why-intro, .why-proof, ' +
        '.acceleration-grid article, .comparison-only, .example-card'
    ));

    const revealOrderBySection = new Map();
    revealBlocks.forEach(block => {
        block.classList.add('reveal-block');
        if (block.matches('.method-kicker, h2')) block.classList.add('reveal-heading');
        else if (block.matches('.method-lead, .section-subtitle, .publication-body > p')) block.classList.add('reveal-copy');
        else if (block.matches('.method-equation-card, .consensus-panel')) block.classList.add('reveal-equation');
        else if (block.matches('figure')) block.classList.add('reveal-figure');
        else block.classList.add('reveal-card');

        const section = block.closest('.section');
        const order = revealOrderBySection.get(section) || 0;
        block.style.setProperty('--reveal-delay', `${Math.min(order, 5) * 90}ms`);
        revealOrderBySection.set(section, order + 1);
    });
    if (reducedMotion || !('IntersectionObserver' in window)) {
        revealBlocks.forEach(block => block.classList.add('is-visible'));
    } else {
        const revealObserver = new IntersectionObserver(function(entries, observer) {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            });
        }, { rootMargin: '0px 0px -10% 0px', threshold: 0.12 });
        revealBlocks.forEach(block => revealObserver.observe(block));
    }

    const pimpcPlayer = document.querySelector('[data-pimpc-player]');
    if (pimpcPlayer) {
        const phases = [
            {
                stage: 'Primal',
                vars: 'Δu, x̄',
                shared: 'Local kernels are running',
                status: 'All horizon stages solve their primal updates together.',
                detail: 'Cached Jₖ, H_A, and H_AN keep every solve stage-local.'
            },
            {
                stage: 'Project / copy',
                vars: 'z, v',
                shared: 'Waiting for local kernels',
                status: 'Each stage projects its constrained copy and reconciles local dynamics.',
                detail: 'The joint state-input constraint is enforced locally; no horizon sweep is needed.'
            },
            {
                stage: 'Dual',
                vars: 'θ, β, λ',
                shared: 'Waiting for local kernels',
                status: 'Dual variables record the three local consensus residuals in parallel.',
                detail: 'Every stage updates the same three equality multipliers at the same time.'
            },
            {
                stage: 'Complete',
                vars: 'stage outputs ready',
                shared: 'Global residual reduction',
                status: 'After all stage kernels finish, lightweight shared work forms the next frozen iterate.',
                detail: 'A sum reduction measures consensus, then Nesterov extrapolates all stage columns together.'
            }
        ];
        const tabs = Array.from(pimpcPlayer.querySelectorAll('[data-pimpc-phase]'));
        const timeline = pimpcPlayer.querySelector('.pimpc-player-timeline');
        const playButton = pimpcPlayer.querySelector('.pimpc-play-toggle');
        const status = pimpcPlayer.querySelector('[data-pimpc-status]');
        const detail = pimpcPlayer.querySelector('[data-pimpc-detail]');
        const shared = pimpcPlayer.querySelector('[data-pimpc-shared]');
        const stageLabels = Array.from(pimpcPlayer.querySelectorAll('[data-pimpc-stage]'));
        const stageVars = Array.from(pimpcPlayer.querySelectorAll('[data-pimpc-vars]'));
        const stages = Array.from(pimpcPlayer.querySelectorAll('.pimpc-stage'));
        let activePhase = 0;
        let playing = !reducedMotion;
        let timer = null;

        function setPhase(index) {
            activePhase = Math.max(0, Math.min(phases.length - 1, index));
            const phase = phases[activePhase];
            pimpcPlayer.dataset.phase = String(activePhase);
            timeline.value = String(activePhase);
            tabs.forEach((tab, tabIndex) => tab.setAttribute('aria-selected', String(tabIndex === activePhase)));
            stageLabels.forEach(label => { label.textContent = phase.stage; });
            stageVars.forEach(label => { label.textContent = phase.vars; });
            stages.forEach(stage => stage.classList.toggle('is-active', activePhase < 3));
            status.textContent = phase.status;
            detail.textContent = phase.detail;
            shared.textContent = phase.shared;
        }

        function updatePlayState() {
            pimpcPlayer.classList.toggle('is-playing', playing);
            playButton.setAttribute('aria-pressed', String(playing));
            playButton.setAttribute('aria-label', playing ? 'Pause iteration animation' : 'Play iteration animation');
            playButton.querySelector('span').textContent = playing ? 'Ⅱ' : '▶';
            window.clearInterval(timer);
            timer = playing ? window.setInterval(() => setPhase((activePhase + 1) % phases.length), 2100) : null;
        }

        tabs.forEach((tab, index) => {
            tab.addEventListener('click', () => setPhase(index));
        });
        timeline.addEventListener('input', event => setPhase(Number(event.target.value)));
        playButton.addEventListener('click', () => {
            playing = !playing;
            updatePlayState();
        });
        if ('IntersectionObserver' in window) {
            const playerObserver = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (!entry.isIntersecting && playing) {
                        window.clearInterval(timer);
                        timer = null;
                    } else if (entry.isIntersecting && playing && !timer) {
                        updatePlayState();
                    }
                });
            }, { threshold: 0.2 });
            playerObserver.observe(pimpcPlayer);
        }
        setPhase(0);
        updatePlayState();
    }

    const ambient = document.querySelector('.ambient-layer');
    if (!reducedMotion && ambient) {
        const starOne = ambient.querySelector('.star-one');
        const starTwo = ambient.querySelector('.star-two');
        const arrows = ambient.querySelector('.ambient-arrow');
        const rocket = ambient.querySelector('.ambient-rocket');
        let ticking = false;

        function updateAmbient() {
            const y = window.scrollY;
            starOne.style.transform = `translate3d(${Math.sin(y / 260) * 16}px, ${y * 0.025}px, 0) rotate(${y * 0.025}deg)`;
            starTwo.style.transform = `translate3d(${Math.cos(y / 300) * 18}px, ${-y * 0.018}px, 0) rotate(${-y * 0.02}deg)`;
            arrows.style.transform = `translate3d(${(y * 0.045) % 150}px, 0, 0)`;
            rocket.style.transform = `translate3d(${-((y * 0.035) % 120)}px, ${Math.sin(y / 180) * 12}px, 0) rotate(-12deg)`;
            ticking = false;
        }

        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(updateAmbient);
                ticking = true;
            }
        }, { passive: true });
        updateAmbient();
    }
});
